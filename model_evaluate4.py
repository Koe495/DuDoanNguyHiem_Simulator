import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Cài đặt font và style cho biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_preprocess_data(filepath):
    print(f"Đang tải dữ liệu từ {filepath}...")
    df = pd.read_csv(filepath)

    # Feature Engineering (Phải khớp 100% với lúc train)
    df = df.sort_values(by=['user_id', 'timestamp'])
    df['hr_baseline'] = df.groupby('user_id')['hr'].transform('mean')
    df['hr_diff_from_baseline'] = df['hr'] - df['hr_baseline']
    df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(lambda x: x.rolling(window=12, min_periods=1).std())
    df['hr_rolling_std'] = df['hr_rolling_std'].fillna(0)

    features = [
        'hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv',
        'body_temp', 'env_temp', 'humidity', 'pressure',
        'oxygen', 'uv_index', 'aqi', 'accel_mag', 'fall_duration'
    ]

    X = df[features]
    y = df['label']

    return X, y, features


def plot_class_distribution(y):
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=y, hue=y, palette=['#2ecc71', '#f1c40f', '#e74c3c'], legend=False)
    plt.title('Phân bố nhãn dữ liệu (Dataset Distribution)', fontsize=14, fontweight='bold')
    plt.xlabel('Nhãn (0: An toàn, 1: Cảnh báo ngầm, 2: Nguy kịch)', fontsize=12)
    plt.ylabel('Số lượng mẫu', fontsize=12)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig('1_class_distribution_pkl.png')
    plt.show(block=False)
    plt.pause(1)  # Hiển thị nhanh rồi đi tiếp


def evaluate_loaded_model(X, y, features, model_path='rf_model.pkl'):
    print(f"\nĐang tải mô hình từ {model_path}...")
    try:
        rf_model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {model_path}. Hãy chạy train_model.py trước.")
        return

    print("Đang chạy Inference (dự đoán) trên dữ liệu...")
    y_pred = rf_model.predict(X)
    y_proba = rf_model.predict_proba(X)

    # ---------------------------------------------------------
    # 1. CLASSIFICATION REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"BÁO CÁO PHÂN LOẠI CỦA MÔ HÌNH {model_path}")
    print("=" * 50)
    target_names = ['0 (An toàn)', '1 (Cảnh báo ngầm)', '2 (Nguy kịch)']
    print(classification_report(y, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 2. CONFUSION MATRIX
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Ma trận nhầm lẫn (Đánh giá file .pkl)', fontsize=14, fontweight='bold')
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)
    plt.tight_layout()
    plt.savefig('2_confusion_matrix_pkl.png')
    plt.show(block=False)
    plt.pause(1)

    # ---------------------------------------------------------
    # 3. FEATURE IMPORTANCE
    # ---------------------------------------------------------
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=[features[i] for i in indices], hue=[features[i] for i in indices],
                palette="viridis", legend=False)
    plt.title('Tầm quan trọng của Đặc trưng (Theo file .pkl)', fontsize=14, fontweight='bold')
    plt.xlabel('Mức độ quan trọng', fontsize=12)
    plt.ylabel('Đặc trưng', fontsize=12)
    plt.tight_layout()
    plt.savefig('3_feature_importance_pkl.png')
    plt.show(block=False)
    plt.pause(1)

    # ---------------------------------------------------------
    # 4. MULTI-CLASS ROC CURVE
    # ---------------------------------------------------------
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(['green', 'orange', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (FPR)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính thật (TPR)', fontsize=12)
    plt.title('Đường cong ROC Đa lớp', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('4_roc_curve_pkl.png')

    print("\n✅ Đã lưu và hiển thị tất cả biểu đồ (có hậu tố _pkl.png)!")
    plt.show()  # Dừng lại ở biểu đồ cuối cùng để người dùng xem


if __name__ == "__main__":
    filepath = 'simulation_survival_dataset5.csv'
    modelpath = 'rf_model_v5.pkl'

    try:
        X, y, features = load_and_preprocess_data(filepath)
        plot_class_distribution(y)
        evaluate_loaded_model(X, y, features, modelpath)
    except Exception as e:
        print(f"❌ Lỗi thực thi: {e}")