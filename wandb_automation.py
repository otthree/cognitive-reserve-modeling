import wandb
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json

# ============================================================
# CONFIG
# ============================================================

WANDB_ENTITY  = "dlee23-university-of-washington"
WANDB_PROJECT = "DivNetFusion AD Classification"

GSHEET_CREDENTIALS_PATH = "service_account.json" 
GSHEET_SPREADSHEET_NAME = "results-hyperparam"  
GSHEET_WORKSHEET_NAME   = "Sheet2"                  

METRIC_KEYS = [
    "train_acc",       # Train ACC
    "test_acc",        # Test ACC
    "best_val_acc",    # Best Val ACC
    "auroc",           # AUROC
    "f1",              # F1
    "best_epoch",      # Stop Epoch
]

CONFIG_KEYS = [
    "model_name",           # Model Name
    "data_modality",        # Data Modality (2 or 3 way)
    "input_dim",            # Input Dim
    "input_resolution",     # Input Resolution
    "patch_size",           # Patch / Tubelet Size
    "embed_dim",            # Embed Dim
    "input_variables",      # Input Variables
    "class_ratio",          # class ratio
    "augmentation",         # Augmentation
    "batch_size",           # Batch
    "lr",                   # LR
    "optimizer",            # Optimizer
    "regularization",       # Regularization
    "scheduler",            # Scheduler
]


HEADERS = [
    "Who", "When", "Exp ID", "Model Name", "Data Modality (2 or 3 way)",
    "Train ACC", "Test ACC", "Best Val ACC", "AUROC", "F1", "Stop Epoch",
    "Input Dim", "Input Resolution", "Patch / Tubelet Size", "Embed Dim",
    "Input Variables", "class ratio", "Augmentation", "Batch", "LR",
    "Optimizer", "Regularization", "Scheduler",
    "WandB URL", "Status"
]



def get_sheets_client():
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(GSHEET_CREDENTIALS_PATH, scopes=scopes)
    return gspread.authorize(creds)


def get_or_create_worksheet(spreadsheet):
    try:
        ws = spreadsheet.worksheet(GSHEET_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=GSHEET_WORKSHEET_NAME, rows=1000, cols=30)
        ws.append_row(HEADERS)
        print(f"새 시트 '{GSHEET_WORKSHEET_NAME}' 생성 및 헤더 추가 완료")
    return ws


def get_existing_exp_ids(worksheet):
    """이미 시트에 있는 Exp ID 목록 (중복 방지)"""
    try:
        col_idx = HEADERS.index("Exp ID") + 1
        values = worksheet.col_values(col_idx)
        return set(values[1:])  # 헤더 제외
    except Exception:
        return set()


def run_to_row(run):
    """WandB run 객체 → 시트 한 행으로 변환"""
    config  = run.config or {}
    summary = run.summary or {}

    # 중첩 딕셔너리 헬퍼
    def cfg(key):
        return config.get(key, "")

    def cfg_nested(section, key):
        return config.get(section, {}).get(key, "")

    def met(key):
        val = summary.get(key, "")
        if isinstance(val, float):
            return round(val, 4)
        return val

    row = [
        run.entity,                                             # Who
        run.created_at[:10] if run.created_at else "",         # When (YYYY-MM-DD)
        run.id,                                                 # Exp ID
        cfg("model_name"),                                      # Model Name 
        cfg("data_modality"),                                   # Data Modality
        met("train_acc"),                                       # Train ACC
        met("val_acc"),                                         # Test ACC → val_acc 사용
        met("val_balanced_acc"),                                # Best Val ACC → val_balanced_acc
        met("val_macro_auc"),                                   # AUROC → val_macro_auc
        met("f1"),                                              # F1 
        met("epoch"),                                           # Stop Epoch
        cfg("input_dim"),                                       # Input Dim
        cfg("input_resolution"),                                # Input Resolution
        cfg_nested("model", "patch_size"),                      # Patch / Tubelet Size
        cfg_nested("model", "embed_dim"),                       # Embed Dim
        cfg("input_variables"),                                 # Input Variables
        cfg("class_ratio"),                                     # class ratio
        cfg("augmentation"),                                    # Augmentation
        cfg_nested("data", "batch_size"),                       # Batch
        cfg_nested("training", "lr"),                           # LR
        cfg_nested("training", "optimizer"),                    # Optimizer
        cfg_nested("training", "weight_decay"),                 # Regularization (weight_decay)
        str(cfg_nested("training", "lr_milestones")),           # Scheduler (milestones)
        run.url,                                                # WandB URL
        run.state,                                              # Status
    ]
    return row


def sync():
    print(f"WandB 프로젝트 '{WANDB_ENTITY}/{WANDB_PROJECT}' 읽는 중...")
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

    print("Google Sheets 연결 중...")
    gc = get_sheets_client()
    spreadsheet = gc.open(GSHEET_SPREADSHEET_NAME)
    worksheet = get_or_create_worksheet(spreadsheet)

    existing_ids = get_existing_exp_ids(worksheet)
    print(f"기존 기록된 run: {len(existing_ids)}개")

    new_rows = []
    updated = 0

    for run in runs:
        if run.id in existing_ids:
            # update_existing_row(worksheet, run)
            continue
        row = run_to_row(run)
        new_rows.append(row)

    if new_rows:
        worksheet.append_rows(new_rows, value_input_option="USER_ENTERED")
        print(f"{len(new_rows)}개 새 run 추가 완료!")
    else:
        print("새로 추가할 run 없음 (모두 최신 상태)")

    print(f"spreadsheet: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")


if __name__ == "__main__":
    sync()