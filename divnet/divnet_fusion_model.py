"""
DivNetFusion: Late fusion of DivNet (3D CNN) + tabular MLP for AD classification.

Architecture:
    CNN branch:  DivNet blocks → flatten(1728) → FC(512) → FC(100) → 100-dim features
    Tab branch:  4 tab features → FC(64) → FC(32) → 32-dim features
    Fusion head: concat(132) → FC(num_classes)
"""

import torch
import torch.nn as nn


class DivNetFusion(nn.Module):
    """
    Late-fusion model combining DivNet 3D CNN backbone with a tabular MLP.

    Args:
        num_filters:     Number of CNN filters (default 64)
        tab_input_dim:   Number of tabular input features (default 4)
        mlp_hidden:      Hidden dim of tabular MLP output (default 32)
        num_classes:     Number of output classes (default 3)
        dropout1:        Dropout after first FC in CNN head (default 0.5)
        dropout2:        Dropout after second FC in CNN head (default 0.3)
        tab_dropout:     Dropout in tabular MLP (default 0.3)
    """

    def __init__(
        self,
        num_filters: int = 64,
        tab_input_dim: int = 2,
        mlp_hidden: int = 32,
        num_classes: int = 3,
        dropout1: float = 0.5,
        dropout2: float = 0.3,
        tab_dropout: float = 0.3,
    ):
        super().__init__()

        # ---- CNN Backbone (DivNet blocks, identical to divnet_model.py) ----
        # Spatial flow: 192 -> 96 -> 24 -> 6 -> 3
        self.block1 = nn.Sequential(
            nn.Conv3d(1, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(num_filters, num_filters, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm3d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # ---- CNN feature head (outputs 100-dim penultimate features) ----
        flatten_size = num_filters * 3 * 3 * 3  # 1728
        self.cnn_head = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout1),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
        )
        cnn_feat_dim = 100

        # ---- Tabular MLP branch (outputs mlp_hidden-dim features) ----
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=tab_dropout),
            nn.Linear(64, mlp_hidden),
            nn.ReLU(inplace=True),
        )

        # ---- Fusion classifier ----
        fusion_dim = cnn_feat_dim + mlp_hidden  # 100 + 32 = 132
        self.fusion_head = nn.Linear(fusion_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def load_cnn_backbone(self, checkpoint_path: str, device):
        """Load DivNet CNN weights (blocks + cnn_head) from a pretrained checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]

        # Map DivNet classifier weights -> cnn_head weights
        # DivNet classifier: [0]=FC(1728->512), [2]=Dropout, [3]=FC(512->100),
        #                    [4]=ReLU,          [5]=Dropout, [6]=FC(100->3)
        # cnn_head:          [0]=FC(1728->512), [2]=Dropout, [3]=FC(512->100),
        #                    [4]=ReLU,          [5]=Dropout
        remap = {}
        for k, v in state.items():
            if k.startswith("block"):
                remap[k] = v
            elif k.startswith("classifier.0") or k.startswith("classifier.3"):
                # FC layers before the final output layer
                new_key = k.replace("classifier.", "cnn_head.")
                remap[new_key] = v

        missing, unexpected = self.load_state_dict(remap, strict=False)
        print(f"Loaded CNN backbone. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    def forward(self, volume: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: [B, 1, 192, 192, 192] MRI tensor
            tab:    [B, 4] normalized tabular features

        Returns:
            logits: [B, num_classes]
        """
        # CNN path
        x = self.block1(volume)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        cnn_feat = self.cnn_head(x)  # [B, 100]

        # Tabular MLP path
        tab_feat = self.tab_mlp(tab)  # [B, 32]

        # Fusion
        fused = torch.cat([cnn_feat, tab_feat], dim=1)  # [B, 132]
        return self.fusion_head(fused)  # [B, 3]


if __name__ == "__main__":
    model = DivNetFusion()
    dummy_vol = torch.randn(2, 1, 192, 192, 192)
    dummy_tab = torch.randn(2, 4)
    out = model(dummy_vol, dummy_tab)
    print(f"Input (MRI):  {dummy_vol.shape}")
    print(f"Input (tab):  {dummy_tab.shape}")
    print(f"Output:       {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
