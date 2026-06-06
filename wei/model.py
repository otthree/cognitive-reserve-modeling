"""
3D Decoupling AD Prediction Network
Wei et al., "A 3D decoupling Alzheimer's disease prediction network based on structural MRI"
Health Information Science and Systems (2025)

Architecture:
  Stem → Stage1 (MSD×4, no multi-scale) → Stage2 (MSD×4, kernels (1,3,5)+(1,3))
       → Stage3 (MSD×6, same as Stage2) → Stage4 (MSD×4, no multi-scale)
       → SA Block → Global AvgPool → FC classifier
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conv_bn_relu(in_ch, out_ch, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Group Convolution Decoupling
# ---------------------------------------------------------------------------

class GroupDecoupling(nn.Module):
    """
    3 parallel group convolutions (G=1, G=2, G=4) with 1×1×1 kernels,
    concatenated and compressed back to out_ch via 1×1×1 conv.

    Requirement: in_ch must be divisible by 4.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        assert in_ch % 4 == 0, f"GroupDecoupling requires in_ch % 4 == 0, got {in_ch}"
        self.gc1 = nn.Conv3d(in_ch, in_ch, 1, groups=1, bias=False)
        self.gc2 = nn.Conv3d(in_ch, in_ch, 1, groups=2, bias=False)
        self.gc4 = nn.Conv3d(in_ch, in_ch, 1, groups=4, bias=False)
        self.compress = nn.Sequential(
            nn.Conv3d(in_ch * 3, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(torch.cat([self.gc1(x), self.gc2(x), self.gc4(x)], dim=1))


# ---------------------------------------------------------------------------
# Multi-Scale Decoupling Block
# ---------------------------------------------------------------------------

class MSDBlock(nn.Module):
    """
    Multi-Scale Decoupling Block.

    Multi-scale mode (len(kernels) > 1):
        - The 3D volume is presented in axial / coronal / sagittal views
          (dimension permutations of a cubic volume) to three separate
          convolutions with kernel sizes given by `kernels`.
        - Feature maps are concatenated along the channel axis.

    Single-conv mode (len(kernels) == 1):
        - Standard 3D conv with kernel kernels[0] (used in Stage 1 & 4).

    Both modes are followed by GroupDecoupling and a residual connection.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 kernels: tuple = (1, 3, 5)):
        super().__init__()
        self.multiscale = len(kernels) > 1

        if self.multiscale:
            n = len(kernels)
            branch_ch = out_ch // n
            self.ms_convs = nn.ModuleList()
            for i, k in enumerate(kernels):
                ch = branch_ch if i < n - 1 else out_ch - branch_ch * (n - 1)
                self.ms_convs.append(nn.Sequential(
                    nn.Conv3d(in_ch, ch, k, stride=stride, padding=k // 2, bias=False),
                    nn.BatchNorm3d(ch),
                    nn.ReLU(inplace=True),
                ))
        else:
            k = kernels[0]
            self.single_conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.decouple = GroupDecoupling(out_ch, out_ch)

        # Shortcut to match dimensions for residual addition
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        if self.multiscale:
            # 3 views: axial (original), coronal (D↔H), sagittal (D↔W)
            # For cubic inputs these all share the same spatial shape
            views = [
                x,
                x.permute(0, 1, 3, 2, 4).contiguous(),   # coronal
                x.permute(0, 1, 4, 2, 3).contiguous(),   # sagittal
            ]
            out = torch.cat([conv(v) for conv, v in zip(self.ms_convs, views)], dim=1)
        else:
            out = self.single_conv(x)

        out = self.decouple(out)
        return out + residual


# ---------------------------------------------------------------------------
# Self-Attention Block
# ---------------------------------------------------------------------------

class SABlock(nn.Module):
    """
    Self-Attention Block operating on a 3D feature map.

    Spatial locations are treated as tokens and processed by multi-head
    self-attention followed by a feed-forward layer (standard Transformer
    encoder block).
    """

    def __init__(self, in_ch: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.attn = nn.MultiheadAttention(in_ch, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(in_ch)
        self.ff = nn.Sequential(
            nn.Linear(in_ch, in_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * 2, in_ch),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, D*H*W, C)

        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out

        tokens = tokens + self.ff(self.norm2(tokens))

        return tokens.permute(0, 2, 1).view(B, C, D, H, W)


# ---------------------------------------------------------------------------
# Full Network
# ---------------------------------------------------------------------------

class DecouplingADNet(nn.Module):
    """
    3D Decoupling AD Prediction Network.

    Stage channels (with base_ch=64):
      Stem :  1 → 32 → 64
      Stage1: 64  → 128   (MSD×4, kernels=(3,),       stride-2 on first block)
      Stage2: 128 → 256   (MSD×4, kernels=(1,3,5)/(1,3), stride-2)
      Stage3: 256 → 512   (MSD×6, kernels=(1,3,5)/(1,3), stride-2)
      Stage4: 512 → 1024  (MSD×4, kernels=(3,),       stride-1)
      SA Block + Global AvgPool + FC

    With input (B, 1, 96, 96, 96):
      After stem  : (B,  64, 24, 24, 24)
      After stage1: (B, 128, 12, 12, 12)
      After stage2: (B, 256,  6,  6,  6)
      After stage3: (B, 512,  3,  3,  3)
      After stage4: (B,1024,  3,  3,  3)
    """

    def __init__(self, num_classes: int = 2, base_ch: int = 64,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        ch = base_ch

        # Stem: two 3D convolutions (32 ch → 64 ch) with stride 2 each
        self.stem = nn.Sequential(
            conv_bn_relu(1, ch // 2, 3, stride=2, padding=1),
            conv_bn_relu(ch // 2, ch, 3, stride=2, padding=1),
        )

        # Stage 1: no multi-scale (like VGG-style basic blocks)
        self.stage1 = self._make_stage(
            in_ch=ch, out_ch=ch * 2, n_blocks=4, first_stride=2,
            kernels_list=[(3,)] * 4,
        )

        # Stage 2: multi-scale, first half (1,3,5), second half (1,3)
        self.stage2 = self._make_stage(
            in_ch=ch * 2, out_ch=ch * 4, n_blocks=4, first_stride=2,
            kernels_list=[(1, 3, 5), (1, 3, 5), (1, 3), (1, 3)],
        )

        # Stage 3: same as stage 2 but 6 blocks
        self.stage3 = self._make_stage(
            in_ch=ch * 4, out_ch=ch * 8, n_blocks=6, first_stride=2,
            kernels_list=[(1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3), (1, 3), (1, 3)],
        )

        # Stage 4: mirrors stage 1 (no multi-scale), no spatial downsampling
        self.stage4 = self._make_stage(
            in_ch=ch * 8, out_ch=ch * 16, n_blocks=4, first_stride=1,
            kernels_list=[(3,)] * 4,
        )

        # Self-Attention Block
        self.sa = SABlock(ch * 16, num_heads=num_heads, dropout=dropout)

        # Classification head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(ch * 16, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, first_stride, kernels_list):
        assert len(kernels_list) == n_blocks
        blocks = [MSDBlock(in_ch, out_ch, stride=first_stride, kernels=kernels_list[0])]
        for i in range(1, n_blocks):
            blocks.append(MSDBlock(out_ch, out_ch, stride=1, kernels=kernels_list[i]))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits : (B, num_classes)
            feat   : (B, base_ch*16)  — used for clustering loss
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.sa(x)
        feat = self.pool(x).flatten(1)
        logits = self.classifier(feat)
        return logits, feat
