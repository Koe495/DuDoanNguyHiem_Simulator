# train8.py
"""
Training Pipeline v8 – Time-Domain Feature Extraction + Random Forest
======================================================================
Dataset  : simulation_survival_dataset8.csv  (genData8.py)
Output   : rf_model_v9.pkl  (tương thích game7.py)

Cải tiến so với v7:
  - Đặt tên file đúng phiên bản (train8 / dataset8)
  - max_depth=25, min_samples_leaf=5 → giảm overfitting so với max_depth=None
  - StratifiedKFold(5) cross-validation trên user-level → ước lượng tin cậy hơn
  - Export feature_importances_ → feature_importance_v9.json

Luồng xử lý:
  1. Đọc CSV
  2. Phân tách user_id thành train/test (stratified, 80/20)
  3. Sliding-window features: 11 kênh × 4 thống kê + 2 baseline = 46 features
  4. StratifiedKFold(5) trên tập train để đánh giá cross-validated accuracy
  5. Huấn luyện RandomForest trên toàn bộ train set
  6. Đánh giá trên test set + confusion matrix
  7. Lưu: rf_model_v9.pkl, X_test_v9.npy, y_test_v9.npy,
          feature_names_v9.json, feature_importance_v9.json
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── Hằng số ──────────────────────────────────────────────────────────────────
DATASET_PATH         = "simulation_survival_dataset8.csv"
MODEL_OUTPUT         = "rf_model_v9.pkl"
X_TEST_OUTPUT        = "X_test_v9.npy"
Y_TEST_OUTPUT        = "y_test_v9.npy"
FEATURE_NAMES_OUTPUT = "feature_names_v9.json"
FEATURE_IMP_OUTPUT   = "feature_importance_v9.json"

# 11 kênh cảm biến (khớp SENSOR_COLS trong game7.py)
SENSOR_COLS: list[str] = [
    "hr", "hrv", "body_temp", "env_temp",
    "humidity", "pressure", "oxygen", "uv_index",
    "aqi", "accel_mag", "fall_duration",
]
# 2 baseline cá nhân – không đưa vào sliding window
BASELINE_COLS: list[str] = ["base_hr", "base_hrv"]

# Khớp genData8.py (TIME_STEP_S=0.2s) và game7.py (SAMPLE_EVERY=6, FPS=30)
WINDOW_SIZE      = 50    # 50 × 0.2s = 10 giây
STRIDE           = 25    # overlap 50%
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
N_ESTIMATORS     = 200
MAX_DEPTH        = 25    # giới hạn độ sâu → giảm overfitting (cũ: None)
MIN_SAMPLES_LEAF = 5     # ít nhất 5 mẫu mỗi lá            (cũ: 1)
N_FOLDS          = 5     # số fold cross-validation


# ── Trích đặc trưng ──────────────────────────────────────────────────────────

def zero_crossing_rate(signal: np.ndarray) -> float:
    """ZCR trên tín hiệu đã center (định nghĩa chuẩn)."""
    centered  = signal - np.mean(signal)
    crossings = np.sum(np.diff(np.sign(centered)) != 0)
    return float(crossings) / max(len(signal) - 1, 1)


def extract_window_features(window: np.ndarray,
                             col_names: list[str]) -> dict[str, float]:
    """
    4 đặc trưng miền thời gian cho mỗi trong 11 kênh:
      mean, std, rms, zcr  →  44 đặc trưng
    """
    features: dict[str, float] = {}
    for j, col in enumerate(col_names):
        sig = window[:, j].astype(float)
        features[f"{col}_mean"] = float(np.mean(sig))
        features[f"{col}_std"]  = float(np.std(sig))
        features[f"{col}_rms"]  = float(np.sqrt(np.mean(sig ** 2)))
        features[f"{col}_zcr"]  = zero_crossing_rate(sig)
    return features


def make_windows(df: pd.DataFrame,
                 user_list: list,
                 window_size: int,
                 stride: int) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Tạo sliding-window features cho danh sách user_id.
    Mỗi cửa sổ: 44 time-domain + 2 baseline = 46 features.
    Nhãn: majority vote trong cửa sổ (minlength=3 đảm bảo đủ 3 lớp).
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

        base_hr  = float(group["base_hr"].iloc[0])
        base_hrv = float(group["base_hrv"].iloc[0])

        for start in range(0, n - window_size + 1, stride):
            window        = data_arr[start : start + window_size]
            window_labels = label_arr[start : start + window_size]
            label         = int(np.bincount(window_labels, minlength=3).argmax())

            feats = extract_window_features(window, SENSOR_COLS)
            feats["base_hr"]  = base_hr
            feats["base_hrv"] = base_hrv
            X_rows.append(feats)
            y_rows.append(label)

    return pd.DataFrame(X_rows), np.array(y_rows)


# ── Pipeline chính ───────────────────────────────────────────────────────────

def train_and_save() -> None:

    # 1. Đọc dữ liệu ──────────────────────────────────────────────────────────
    print(f"[1/6] Đọc dữ liệu từ '{DATASET_PATH}'...")
    try:
        df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    except FileNotFoundError:
        print(f"  Không tìm thấy '{DATASET_PATH}'. Hãy chạy genData8.py trước.")
        return

    label_dist = df["label"].value_counts().sort_index().to_dict()
    print(f"     Tổng dòng : {len(df):,}")
    print(f"     Phân bổ  : {label_dist}")

    # 2. Split user_id (stratified) ───────────────────────────────────────────
    print(f"\n[2/6] Phân chia train/test theo user_id "
          f"(stratified, test={int(TEST_SIZE*100)}%)...")

    all_users  = df["user_id"].unique().tolist()
    user_label = [int(df[df["user_id"] == u]["label"].mode().iloc[0])
                  for u in all_users]

    train_users, test_users = train_test_split(
        all_users,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = user_label,
    )
    print(f"     Train users: {len(train_users):,}  |  Test users: {len(test_users):,}")

    # 3. Trích đặc trưng ──────────────────────────────────────────────────────
    print(f"\n[3/6] Trích đặc trưng sliding-window "
          f"(window={WINDOW_SIZE}×0.2s={WINDOW_SIZE*0.2:.0f}s, "
          f"stride={STRIDE}×0.2s={STRIDE*0.2:.0f}s)...")

    X_train, y_train = make_windows(df, train_users, WINDOW_SIZE, STRIDE)
    X_test,  y_test  = make_windows(df, test_users,  WINDOW_SIZE, STRIDE)

    print(f"     Train windows : {len(X_train):,}")
    print(f"     Test  windows : {len(X_test):,}")
    print(f"     Features      : {X_train.shape[1]}")
    print(f"     Train labels  : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"     Test  labels  : {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    X_train_np = X_train.values
    X_test_np  = X_test.values

    # 4. Cross-validation trên tập train ──────────────────────────────────────
    print(f"\n[4/6] StratifiedKFold({N_FOLDS}) cross-validation trên tập train...")

    rf_cv = RandomForestClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        class_weight     = "balanced",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(rf_cv, X_train_np, y_train, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    print(f"     CV accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"     Folds      : {[f'{v:.4f}' for v in cv_acc]}")

    # 5. Huấn luyện trên toàn bộ train set ────────────────────────────────────
    print(f"\n[5/6] Huấn luyện RandomForest "
          f"(n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}, class_weight='balanced')...")

    rf = RandomForestClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        class_weight     = "balanced",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    rf.fit(X_train_np, y_train)

    y_pred = rf.predict(X_test_np)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n     Test accuracy: {acc:.4f}")
    print("\n" + classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Caution (1)", "Critical (2)"],
    ))

    print("     Confusion matrix (hàng=thực tế, cột=dự đoán):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"       {'':14s}  Pred 0  Pred 1  Pred 2")
    for i, row in enumerate(cm):
        print(f"       Actual {i}     :  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")

    # 6. Lưu artifacts ────────────────────────────────────────────────────────
    print("\n[6/6] Lưu mô hình và artifacts...")

    feature_names = list(X_train.columns)
    imp_pairs = sorted(
        zip(feature_names, rf.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    feature_importance = {name: float(imp) for name, imp in imp_pairs}

    try:
        joblib.dump(rf, MODEL_OUTPUT)
        np.save(X_TEST_OUTPUT, X_test_np)
        np.save(Y_TEST_OUTPUT, y_test)

        with open(FEATURE_NAMES_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, ensure_ascii=False, indent=2)

        with open(FEATURE_IMP_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(feature_importance, f, ensure_ascii=False, indent=2)

        print(f"     Model      → {MODEL_OUTPUT}")
        print(f"     X_test     → {X_TEST_OUTPUT}")
        print(f"     y_test     → {Y_TEST_OUTPUT}")
        print(f"     Features   → {FEATURE_NAMES_OUTPUT}")
        print(f"     Importance → {FEATURE_IMP_OUTPUT}")

        print("\n     Top 10 features quan trọng nhất:")
        for name, imp in imp_pairs[:10]:
            bar = "█" * int(imp * 400)
            print(f"       {name:25s}  {imp:.4f}  {bar}")

        print("\n✅ Hoàn tất huấn luyện!")

    except Exception as e:
        print(f"❌ Lỗi khi lưu: {e}")


if __name__ == "__main__":
    train_and_save()