# train7.py
"""
Training Pipeline v7 – Time-Domain Feature Extraction + Random Forest
======================================================================
Thay đổi so với phiên bản trước:
  - Split train/test theo user_id (không còn data leakage window-level)
  - WINDOW_SIZE=50, STRIDE=25  → cửa sổ 10s @ 0.2s/step (khớp game6.py)
  - Đọc simulation_survival_dataset6v2.csv (TIME_STEP_S=0.2s)

Luồng xử lý:
  1. Đọc CSV
  2. Phân tách user_id thành train_users / test_users (stratified)
  3. Tạo sliding-window features cho từng tập riêng biệt
  4. Huấn luyện RandomForestClassifier (200 cây, class_weight='balanced')
  5. Lưu: rf_model_v8.pkl, X_test_v8.npy, y_test_v8.npy, feature_names_v8.json
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Hằng số ──────────────────────────────────────────────────────────────────
DATASET_PATH         = "simulation_survival_dataset7.csv"
MODEL_OUTPUT         = "rf_model_v8.pkl"
X_TEST_OUTPUT        = "X_test_v8.npy"
Y_TEST_OUTPUT        = "y_test_v8.npy"
FEATURE_NAMES_OUTPUT = "feature_names_v8.json"

# 11 kênh cảm biến cửa sổ trượt
SENSOR_COLS: list[str] = [
    "hr", "hrv", "body_temp", "env_temp",
    "humidity", "pressure", "oxygen", "uv_index",
    "aqi", "accel_mag", "fall_duration",
]
# 2 cột baseline cá nhân – hằng số mỗi user, không đưa vào sliding window
BASELINE_COLS: list[str] = ["base_hr", "base_hrv"]

# Khớp với TIME_STEP_S=0.2s trong genData7.py và SAMPLE_EVERY=6 trong game6.py
WINDOW_SIZE  = 50   # 50 × 0.2s = 10 giây
STRIDE       = 25   # 25 × 0.2s =  5 giây (overlap 50%)
TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 200


# ── Hàm trích đặc trưng ──────────────────────────────────────────────────────

def zero_crossing_rate(signal: np.ndarray) -> float:
    centered  = signal - np.mean(signal)
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    return float(crossings) / max(len(signal) - 1, 1)


def extract_window_features(window: np.ndarray,
                             col_names: list[str]) -> dict[str, float]:
    """Trích 4 đặc trưng miền thời gian (mean, std, rms, zcr) cho mỗi kênh."""
    features: dict[str, float] = {}
    for j, col in enumerate(col_names):
        sig = window[:, j].astype(float)
        features[f"{col}_mean"] = float(np.mean(sig))
        features[f"{col}_std"]  = float(np.std(sig))
        features[f"{col}_rms"]  = float(np.sqrt(np.mean(sig ** 2)))
        features[f"{col}_zcr"]  = zero_crossing_rate(sig)
    return features


def _make_windows(df: pd.DataFrame,
                  user_list,
                  window_size: int,
                  stride: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Tạo sliding-window features cho danh sách user_id.
    Mỗi cửa sổ gồm 44 đặc trưng time-domain (11 kênh × 4 stats)
    + 2 giá trị baseline cá nhân (base_hr, base_hrv) = 46 features.
    """
    X_rows: list[dict] = []
    y_rows: list[int]  = []

    for uid in user_list:
        group     = (df[df["user_id"] == uid]
                     .sort_values("timestamp")
                     .reset_index(drop=True))
        data_arr  = group[SENSOR_COLS].values
        label_arr = group["label"].values.astype(int)
        n         = len(data_arr)

        # Baseline cá nhân – hằng số trong cả timeline của user
        base_hr  = float(group["base_hr"].iloc[0])
        base_hrv = float(group["base_hrv"].iloc[0])

        for start in range(0, n - window_size + 1, stride):
            window        = data_arr[start : start + window_size]
            window_labels = label_arr[start : start + window_size]
            label         = int(np.bincount(window_labels).argmax())

            feats = extract_window_features(window, SENSOR_COLS)
            feats["base_hr"]  = base_hr    # thêm trực tiếp vào dict feature
            feats["base_hrv"] = base_hrv
            X_rows.append(feats)
            y_rows.append(label)

    return pd.DataFrame(X_rows), np.array(y_rows)


# ── Huấn luyện & lưu mô hình ─────────────────────────────────────────────────

def train_and_save() -> None:
    # 1. Đọc dữ liệu
    print(f"[1/5] Đọc dữ liệu từ '{DATASET_PATH}'...")
    try:
        df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"  Không tìm thấy '{DATASET_PATH}'. Hãy chạy genData6v2.py trước.")
        return

    print(f"     Tổng dòng: {len(df):,}  |  Phân bổ nhãn: "
          f"{df['label'].value_counts().sort_index().to_dict()}")

    # 2. Split user_id (stratified theo nhãn đa số của từng user)
    print(f"[2/5] Phân chia train/test theo user_id "
          f"(stratified, test={int(TEST_SIZE*100)}%)...")

    all_users  = df["user_id"].unique().tolist()   # plain list → sklearn-safe
    user_label = [
        int(df[df["user_id"] == u]["label"].mode().iloc[0])
        for u in all_users
    ]

    train_users, test_users = train_test_split(
        all_users,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = user_label,
    )
    print(f"     Train users: {len(train_users):,}  |  Test users: {len(test_users):,}")

    # 3. Tạo sliding-window features cho từng tập
    print(f"[3/5] Trích đặc trưng Time-Domain "
          f"(window={WINDOW_SIZE}×0.2s={WINDOW_SIZE*0.2:.0f}s, "
          f"stride={STRIDE}×0.2s={STRIDE*0.2:.0f}s)...")

    X_train, y_train = _make_windows(df, train_users, WINDOW_SIZE, STRIDE)
    X_test,  y_test  = _make_windows(df, test_users,  WINDOW_SIZE, STRIDE)

    print(f"     Train windows: {len(X_train):,}  |  "
          f"Test windows: {len(X_test):,}  |  "
          f"Features: {X_train.shape[1]}")

    # 4. Huấn luyện Random Forest
    print(f"[4/5] Huấn luyện RandomForest "
          f"(n_estimators={N_ESTIMATORS}, class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators = N_ESTIMATORS,
        max_depth    = None,
        class_weight = "balanced",
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )
    rf.fit(X_train.values, y_train)   # numpy → không gắn feature names vào model

    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n     Accuracy trên tập test: {acc:.4f}")
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Caution (1)", "Critical (2)"],
    ))

    # 5. Lưu artifact
    print("[5/5] Lưu mô hình và dữ liệu test...")
    try:
        joblib.dump(rf, MODEL_OUTPUT)
        np.save(X_TEST_OUTPUT, X_test.values)
        np.save(Y_TEST_OUTPUT, y_test)
        with open(FEATURE_NAMES_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)

        print(f"     Mô hình   → {MODEL_OUTPUT}")
        print(f"     X_test    → {X_TEST_OUTPUT}")
        print(f"     y_test    → {Y_TEST_OUTPUT}")
        print(f"     Features  → {FEATURE_NAMES_OUTPUT}")
        print("\n✅ Hoàn tất huấn luyện!")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file: {e}")


if __name__ == "__main__":
    train_and_save()
