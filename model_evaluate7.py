# model_evaluate7.py
"""
Model Evaluation v7 – RandomForest Survival Risk Classifier
============================================================
Đánh giá toàn diện mô hình rf_model_v8.pkl.

Chỉ số & biểu đồ:
  [1] Classification Report (Accuracy, Precision, Recall, F1 per class)
  [2] ROC-AUC (One-vs-Rest, macro)
  [3] Confusion Matrix          → confusion_matrix_v7.png
  [4] Feature Importance        → feature_importance_v7.png
  [5] ROC Curves                → roc_curves_v7.png
  [6] Precision-Recall Curves   → pr_curves_v7.png
  [7] Per-Scenario Accuracy     → scenario_accuracy_v7.png  (nếu CSV tồn tại)
  [8] Confidence Distribution   → confidence_hist_v7.png

Yêu cầu: chạy train7.py trước để có rf_model_v8.pkl, X_test_v8.npy,
          y_test_v8.npy, feature_names_v8.json.
"""

import json
import sys
import re
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Đảm bảo stdout hiển thị được tiếng Việt trên Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

# ── Đường dẫn artefact ───────────────────────────────────────────────────────
MODEL_PATH         = "rf_model_v8.pkl"
X_TEST_PATH        = "X_test_v8.npy"
Y_TEST_PATH        = "y_test_v8.npy"
FEATURE_NAMES_PATH = "feature_names_v8.json"
DATASET_PATH       = "simulation_survival_dataset6v2.csv"
SCENARIOS_JSON     = "scenarios.json"

# Hằng số train phải khớp với train7.py
WINDOW_SIZE  = 50
STRIDE       = 25
SENSOR_COLS  = [
    "hr", "hrv", "body_temp", "env_temp",
    "humidity", "pressure", "oxygen", "uv_index",
    "aqi", "accel_mag", "fall_duration",
]

CLASS_NAMES = ["Safe (0)", "Caution (1)", "Critical (2)"]
COLORS      = ["#2ecc71", "#f39c12", "#e74c3c"]
TOP_N       = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  Nạp artefact
# ═══════════════════════════════════════════════════════════════════════════════

def load_artefacts():
    required = {
        "Mô hình"      : MODEL_PATH,
        "X_test"       : X_TEST_PATH,
        "y_test"       : Y_TEST_PATH,
        "Feature names": FEATURE_NAMES_PATH,
    }
    for label, path in required.items():
        if not Path(path).exists():
            print(f"  Không tìm thấy {label}: '{path}'")
            print("  → Hãy chạy train7.py trước.")
            sys.exit(1)

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    with open(FEATURE_NAMES_PATH, encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    return model, X_test, y_test, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
#  [1][2] Chỉ số phân loại
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics(y_true, y_pred, y_prob, model_info: str = MODEL_PATH) -> None:
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    print("=" * 65)
    print(f"  ĐÁNH GIÁ MÔ HÌNH: {model_info}")
    print("=" * 65)
    print(f"\n  Accuracy       : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  ROC-AUC (macro): {roc:.4f}  (One-vs-Rest)")

    # Per-class AP
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    aps = [
        average_precision_score(y_bin[:, i], y_prob[:, i])
        for i in range(3)
    ]
    print(f"\n  Average Precision per class:")
    for i, (cn, ap) in enumerate(zip(CLASS_NAMES, aps)):
        print(f"    {cn:18s}: {ap:.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES)}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
#  [3] Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred,
                          out: str = "confusion_matrix_v7.png") -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=axes[0], cmap="Blues", colorbar=True, values_format="d"
    )
    axes[0].set_title("Confusion Matrix – Số lượng tuyệt đối", fontsize=11)

    # Normalized (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(cm_norm, display_labels=CLASS_NAMES).plot(
        ax=axes[1], cmap="Greens", colorbar=True, values_format=".2f"
    )
    axes[1].set_title("Confusion Matrix – Tỷ lệ (Recall per class)", fontsize=11)

    plt.suptitle(f"Confusion Matrix – {MODEL_PATH}", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, out)


# ═══════════════════════════════════════════════════════════════════════════════
#  [4] Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(model, feature_names: list[str],
                            out: str = "feature_importance_v7.png") -> None:
    imps = model.feature_importances_
    idx  = np.argsort(imps)[::-1][:TOP_N]
    names = [feature_names[i] for i in idx]
    vals  = imps[idx]

    # Color by sensor channel
    channel_colors = {
        "hr": "#e74c3c", "hrv": "#e67e22", "body_temp": "#f1c40f",
        "env_temp": "#2ecc71", "humidity": "#1abc9c", "pressure": "#3498db",
        "oxygen": "#9b59b6", "uv_index": "#e91e63", "aqi": "#795548",
        "accel_mag": "#607d8b", "fall_duration": "#ff5722",
    }
    bar_colors = [
        channel_colors.get(n.split("_mean")[0].split("_std")[0]
                           .split("_rms")[0].split("_zcr")[0], "#95a5a6")
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, TOP_N * 0.40)))
    bars = ax.barh(range(TOP_N), vals[::-1], color=bar_colors[::-1],
                   edgecolor="white", height=0.7)
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Độ quan trọng (Gini Impurity Decrease)", fontsize=10)
    ax.set_title(f"Top {TOP_N} Đặc trưng Quan trọng – {MODEL_PATH}", fontsize=12)

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)

    # Legend for channels
    seen = {}
    for n, c in zip(names, bar_colors):
        ch = n.rsplit("_", 1)[0] if "_" in n else n
        seen[ch] = c
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in list(seen.values())[:8]]
    labels  = list(seen.keys())[:8]
    ax.legend(handles, labels, loc="lower right", fontsize=8,
              title="Kênh cảm biến", title_fontsize=8)

    plt.tight_layout()
    _save(fig, out)


# ═══════════════════════════════════════════════════════════════════════════════
#  [5] ROC Curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(y_true, y_prob,
                   out: str = "roc_curves_v7.png") -> None:
    y_bin     = label_binarize(y_true, classes=[0, 1, 2])
    mean_fpr  = np.linspace(0, 1, 300)
    tpr_accum = np.zeros(300)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_val = auc(fpr, tpr)
        tpr_accum += np.interp(mean_fpr, fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2.2,
                label=f"{CLASS_NAMES[i]}  (AUC={roc_val:.3f})")

    macro_tpr = tpr_accum / 3
    ax.plot(mean_fpr, macro_tpr, color="navy", lw=2.5, linestyle="--",
            label=f"Macro-avg  (AUC={auc(mean_fpr, macro_tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"Đường cong ROC (One-vs-Rest) – {MODEL_PATH}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, out)


# ═══════════════════════════════════════════════════════════════════════════════
#  [6] Precision-Recall Curves
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pr_curves(y_true, y_prob,
                  out: str = "pr_curves_v7.png") -> None:
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(3):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(rec, prec, color=COLORS[i], lw=2.2,
                label=f"{CLASS_NAMES[i]}  (AP={ap:.3f})")

        # Baseline (random classifier at class prevalence)
        prevalence = y_bin[:, i].mean()
        ax.axhline(prevalence, color=COLORS[i], lw=0.8, linestyle=":",
                   alpha=0.5)

    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="Recall", ylabel="Precision",
           title=f"Đường cong Precision-Recall – {MODEL_PATH}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    _save(fig, out)


# ═══════════════════════════════════════════════════════════════════════════════
#  [7] Per-Scenario Accuracy  (cần CSV + scenarios.json)
# ═══════════════════════════════════════════════════════════════════════════════

def _zcr(sig: np.ndarray) -> float:
    c = sig - float(np.mean(sig))
    return float(np.sum(np.diff(np.sign(c)) != 0)) / max(len(sig) - 1, 1)


def _window_features(window: np.ndarray) -> np.ndarray:
    feats = []
    for j in range(window.shape[1]):
        s = window[:, j].astype(float)
        feats += [float(np.mean(s)), float(np.std(s)),
                  float(np.sqrt(np.mean(s ** 2))), _zcr(s)]
    return np.array(feats, dtype=float)


def _extract_scenario_from_uid(uid: str) -> str:
    """
    Trích tên kịch bản từ user_id.
    Định dạng: User_{SCENARIO}_{003}  hoặc  User_X_{A}_{B}_{003}
    """
    if uid.startswith("User_X_"):
        parts = uid.split("_")
        return f"CROSS_{parts[2]}_{parts[3]}"
    m = re.match(r"User_(.+)_\d{3}$", uid)
    return m.group(1) if m else "UNKNOWN"


def plot_scenario_accuracy(model, feature_names: list[str],
                           out: str = "scenario_accuracy_v7.png") -> None:
    if not Path(DATASET_PATH).exists():
        print(f"  (Bỏ qua per-scenario: không tìm thấy '{DATASET_PATH}')")
        return

    try:
        import pandas as pd
    except ImportError:
        print("  (Bỏ qua per-scenario: pandas không khả dụng)")
        return

    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])

    # Build windows per scenario
    scenario_results: dict[str, dict] = {}

    for uid in df["user_id"].unique():
        scenario = _extract_scenario_from_uid(str(uid))
        group = (df[df["user_id"] == uid]
                 .sort_values("timestamp")
                 .reset_index(drop=True))
        data_arr  = group[SENSOR_COLS].values.astype(float)
        label_arr = group["label"].values.astype(int)
        n = len(data_arr)

        preds, trues = [], []
        base_hr  = float(group["base_hr"].iloc[0])  if "base_hr"  in group.columns else 70.0
        base_hrv = float(group["base_hrv"].iloc[0]) if "base_hrv" in group.columns else 62.0

        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            w    = data_arr[start : start + WINDOW_SIZE]
            lbl  = int(np.bincount(label_arr[start : start + WINDOW_SIZE]).argmax())
            # 44 time-domain features + 2 baseline cá nhân = 46 features
            feat = np.append(_window_features(w), [base_hr, base_hrv]).reshape(1, -1)
            pred = int(model.predict(feat)[0])
            preds.append(pred)
            trues.append(lbl)

        if not preds:
            continue
        if scenario not in scenario_results:
            scenario_results[scenario] = {"correct": 0, "total": 0,
                                          "label_target": trues[0]}
        sc = scenario_results[scenario]
        sc["correct"] += sum(p == t for p, t in zip(preds, trues))
        sc["total"]   += len(preds)

    if not scenario_results:
        return

    # Sort by label_target then name
    sorted_names = sorted(scenario_results,
                          key=lambda k: (scenario_results[k]["label_target"], k))
    accs   = [scenario_results[k]["correct"] / scenario_results[k]["total"]
              for k in sorted_names]
    counts = [scenario_results[k]["total"] for k in sorted_names]
    labels_t = [scenario_results[k]["label_target"] for k in sorted_names]

    bar_cols = [COLORS[min(2, int(l))] for l in labels_t]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_names) * 0.9), 6))
    x = np.arange(len(sorted_names))
    bars = ax.bar(x, accs, color=bar_cols, edgecolor="white", width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in sorted_names],
                       fontsize=8, rotation=0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(f"Độ chính xác phân loại theo kịch bản – {MODEL_PATH}",
                 fontsize=12)
    ax.axhline(0.9, color="gray", lw=1, linestyle="--", alpha=0.5,
               label="Ngưỡng 90%")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    for bar, acc, n_win in zip(bars, accs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{acc:.2f}\n(n={n_win})",
                ha="center", va="bottom", fontsize=7)

    # Legend for class colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    ax.legend(handles, CLASS_NAMES, loc="lower left", fontsize=8,
              title="Nhãn mục tiêu", title_fontsize=8)

    plt.tight_layout()
    _save(fig, out)
    print(f"\n  Per-scenario summary:")
    print(f"  {'Kịch bản':<35} {'Acc':>6}  {'Windows':>8}")
    print(f"  {'-'*52}")
    for k, acc, n in zip(sorted_names, accs, counts):
        print(f"  {k:<35} {acc:>6.3f}  {n:>8,}")


# ═══════════════════════════════════════════════════════════════════════════════
#  [8] Confidence Distribution per class
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confidence_hist(y_true, y_pred, y_prob,
                         out: str = "confidence_hist_v7.png") -> None:
    """
    Phân phối xác suất cao nhất (confidence) cho:
      - Dự đoán đúng vs. sai
      - Từng lớp được dự đoán
    """
    conf   = np.max(y_prob, axis=1)
    correct = (y_pred == y_true)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Subplot 1: đúng vs sai ────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(conf[correct],  bins=30, color="#2ecc71", alpha=0.7,
            label=f"Đúng (n={correct.sum():,})", density=True)
    ax.hist(conf[~correct], bins=30, color="#e74c3c", alpha=0.7,
            label=f"Sai  (n={(~correct).sum():,})", density=True)
    ax.axvline(conf[correct].mean(),  color="#27ae60", lw=1.5, linestyle="--",
               label=f"Mean đúng={conf[correct].mean():.3f}")
    ax.axvline(conf[~correct].mean(), color="#c0392b", lw=1.5, linestyle="--",
               label=f"Mean sai={conf[~correct].mean():.3f}")
    ax.set(xlabel="Confidence (max proba)", ylabel="Mật độ",
           title="Phân phối Confidence: Đúng vs. Sai")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ── Subplot 2: per predicted class ───────────────────────────────────────
    ax = axes[1]
    for i, (cn, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = y_pred == i
        if mask.sum() == 0:
            continue
        ax.hist(conf[mask], bins=25, color=col, alpha=0.65,
                label=f"{cn} (n={mask.sum():,})", density=True)
    ax.set(xlabel="Confidence (max proba)", ylabel="Mật độ",
           title="Phân phối Confidence theo Lớp dự đoán")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.suptitle(f"Phân phối Độ tin cậy Dự đoán – {MODEL_PATH}",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, out)

    # Console summary
    print(f"\n  Confidence summary:")
    print(f"    Toàn bộ     : mean={conf.mean():.3f}  std={conf.std():.3f}"
          f"  min={conf.min():.3f}  max={conf.max():.3f}")
    print(f"    Dự đoán đúng: mean={conf[correct].mean():.3f}"
          f"  ({correct.mean()*100:.1f}% tổng)")
    print(f"    Dự đoán sai : mean={conf[~correct].mean():.3f}"
          f"  ({(~correct).mean()*100:.1f}% tổng)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tiện ích lưu file
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: str) -> None:
    try:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"    → {path}")
    except Exception as e:
        print(f"    (Lỗi lưu {path}: {e})")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate() -> None:
    print("\n" + "=" * 65)
    print("  BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH – model_evaluate7.py")
    print("=" * 65 + "\n")

    # Nạp artefact
    model, X_test, y_test, feature_names = load_artefacts()
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Mẫu test: {len(y_test):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Nhãn test phân bổ: "
          f"{dict(zip(*np.unique(y_test, return_counts=True)))}\n")

    # Dự đoán – luôn dùng numpy để khớp với cách train (rf.fit(X.values, y))
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    except Exception as e:
        print(f"  Lỗi khi dự đoán: {e}")
        return

    # [1][2] Chỉ số console
    print_metrics(y_test, y_pred, y_prob)

    # [3]-[8] Biểu đồ
    print("\n  Đang tạo biểu đồ...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_names)
    plot_roc_curves(y_test, y_prob)
    plot_pr_curves(y_test, y_prob)
    plot_confidence_hist(y_test, y_pred, y_prob)

    print("\n  Đang đánh giá per-scenario (có thể mất vài phút)...")
    plot_scenario_accuracy(model, feature_names)

    print("\n  Đánh giá hoàn tất!")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
