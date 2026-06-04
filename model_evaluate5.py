# model_evaluate5.py
"""
Model Evaluation v5 – RandomForest Survival Risk Classifier
============================================================
Đánh giá mô hình rf_model_v7.pkl trên tập kiểm tra đã lưu bởi train7.py.

Các chỉ số & biểu đồ đầu ra:
  • Classification Report  (Accuracy, Precision, Recall, F1-score per class)
  • ROC-AUC (One-vs-Rest, macro average)
  • Confusion Matrix        → confusion_matrix_v5.png
  • Feature Importance      → feature_importance_v5.png
  • ROC Curves              → roc_curves_v5.png
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")   # Render không cần màn hình

# ── Đường dẫn artefact ───────────────────────────────────────────────────────
MODEL_PATH = "rf_model_v8.pkl"
X_TEST_PATH = "X_test_v8.npy"
Y_TEST_PATH = "y_test_v8.npy"
FEATURE_NAMES_PATH = "feature_names_v8.json"

CLASS_NAMES = ["Safe (0)", "Caution (1)", "Critical (2)"]
COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]
TOP_N_FEATURES = 20     # Số đặc trưng quan trọng nhất hiển thị trong biểu đồ


# ── Tiện ích ─────────────────────────────────────────────────────────────────

def load_artefacts() -> tuple:
    """
    Nạp mô hình, tập kiểm tra và danh sách đặc trưng từ đĩa.

    Returns:
        Tuple (model, X_test, y_test, feature_names).

    Raises:
        SystemExit nếu bất kỳ file nào không tồn tại.
    """
    required = {
        "Mô hình": MODEL_PATH,
        "X_test":  X_TEST_PATH,
        "y_test":  Y_TEST_PATH,
        "Feature names": FEATURE_NAMES_PATH,
    }
    for label, path in required.items():
        try:
            open(path).close()
        except FileNotFoundError:
            print(f"❌ Không tìm thấy {label}: '{path}'")
            print("   → Hãy chạy train7.py trước.")
            sys.exit(1)

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)

    return model, X_test, y_test, feature_names


# ── Đánh giá chỉ số ──────────────────────────────────────────────────────────

def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    """
    In Accuracy, Classification Report và ROC-AUC ra console.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        y_prob: Xác suất dự đoán mỗi lớp (n_samples × n_classes).
    """
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    print("=" * 60)
    print("         ĐÁNH GIÁ MÔ HÌNH rf_model_v7.pkl")
    print("=" * 60)
    print(f"\n  Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  ROC-AUC   : {roc:.4f} (One-vs-Rest, macro)\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print("=" * 60)


# ── Biểu đồ ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = "confusion_matrix_v5.png",
) -> None:
    """
    Vẽ và lưu Confusion Matrix dạng heatmap màu.

    Args:
        y_true: Nhãn thực tế.
        y_pred: Nhãn dự đoán.
        output_path: Đường dẫn file ảnh đầu ra.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )
    ax.set_title("Ma trận Nhầm lẫn (Confusion Matrix)\nrf_model_v7.pkl", fontsize=13)
    plt.tight_layout()

    try:
        fig.savefig(output_path, dpi=150)
        print(f"  📊 Confusion Matrix → {output_path}")
    except Exception as e:
        print(f"  ⚠️  Không thể lưu confusion matrix: {e}")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
    output_path: str = "feature_importance_v5.png",
) -> None:
    """
    Vẽ biểu đồ cột nằm ngang thể hiện TOP đặc trưng quan trọng nhất.

    Args:
        model: Mô hình RandomForest đã huấn luyện.
        feature_names: Danh sách tên đặc trưng tương ứng.
        top_n: Số đặc trưng hàng đầu cần hiển thị.
        output_path: Đường dẫn file ảnh đầu ra.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(
        range(top_n), top_vals[::-1],
        color="#3498db", edgecolor="white", height=0.65,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Độ quan trọng (Gini Impurity Decrease)", fontsize=10)
    ax.set_title(
        f"Top {top_n} Đặc trưng Quan trọng nhất\nrf_model_v7.pkl",
        fontsize=13,
    )

    # Gán nhãn giá trị trên mỗi cột
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=8,
        )

    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=150)
        print(f"  📊 Feature Importance → {output_path}")
    except Exception as e:
        print(f"  ⚠️  Không thể lưu feature importance: {e}")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int = 3,
    output_path: str = "roc_curves_v5.png",
) -> None:
    """
    Vẽ đường cong ROC riêng cho từng lớp (One-vs-Rest) và macro-average.

    Args:
        y_true: Nhãn thực tế.
        y_prob: Xác suất dự đoán (n_samples × n_classes).
        n_classes: Số lớp phân loại.
        output_path: Đường dẫn file ảnh đầu ra.
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr_sum = np.zeros(200)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        mean_tpr_sum += interp_tpr
        ax.plot(
            fpr, tpr,
            color=COLORS[i],
            lw=2,
            label=f"{CLASS_NAMES[i]}  (AUC = {roc_auc_val:.3f})",
        )

    mean_tpr = mean_tpr_sum / n_classes
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(
        mean_fpr, mean_tpr,
        color="navy", lw=2.5, linestyle="--",
        label=f"Macro-avg  (AUC = {macro_auc:.3f})",
    )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Đường cong ROC (One-vs-Rest)\nrf_model_v7.pkl", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    try:
        fig.savefig(output_path, dpi=150)
        print(f"  📊 ROC Curves         → {output_path}")
    except Exception as e:
        print(f"  ⚠️  Không thể lưu ROC curves: {e}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def evaluate() -> None:
    """Hàm chính: nạp artefact, tính toán chỉ số và xuất biểu đồ."""
    print("\n=== BẮT ĐẦU ĐÁNH GIÁ MÔ Hình RF ===\n")

    # Nạp artefact
    model, X_test, y_test, feature_names = load_artefacts()
    print(f"  ✅ Nạp mô hình thành công.  Số mẫu test: {len(y_test):,}\n")

    # Dự đoán
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")
        return

    # In chỉ số
    print_classification_metrics(y_test, y_pred, y_prob)

    # Xuất biểu đồ
    print("\n  Đang tạo biểu đồ...")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_names)
    plot_roc_curves(y_test, y_prob)

    print("\n✅ Đánh giá hoàn tất!")


if __name__ == "__main__":
    evaluate()
