import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Cài đặt font và style cho biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_and_preprocess_data(filepath):
    print(f"Đang tải dữ liệu từ {filepath}...")
    df = pd.read_csv(filepath)

    # Feature Engineering (Giống với môi trường Game)
    df = df.sort_values(by=['user_id', 'timestamp'])
    df['hr_baseline'] = df.groupby('user_id')['hr'].transform('mean')
    df['hr_diff_from_baseline'] = df['hr'] - df['hr_baseline']
    df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(lambda x: x.rolling(window=12, min_periods=1).std())
    df['hr_rolling_std'].fillna(0, inplace=True)

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

    # Hiện số liệu trên cột
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig('1_class_distribution.png')
    plt.show()


def evaluate_model(X, y, features):
    # Chia tập Train/Test (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nĐang huấn luyện mô hình Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Dự đoán
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)

    # ---------------------------------------------------------
    # 1. CLASSIFICATION REPORT
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT)")
    print("=" * 50)
    target_names = ['0 (An toàn)', '1 (Cảnh báo ngầm)', '2 (Nguy kịch)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ---------------------------------------------------------
    # 2. CONFUSION MATRIX
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
    plt.ylabel('Nhãn thực tế (True Label)', fontsize=12)
    plt.xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12)
    plt.tight_layout()
    plt.savefig('2_confusion_matrix.png')
    plt.show()

    # ---------------------------------------------------------
    # 3. FEATURE IMPORTANCE (TẦM QUAN TRỌNG ĐẶC TRƯNG)
    # ---------------------------------------------------------
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance[indices], y=[features[i] for i in indices], hue=[features[i] for i in indices],
                palette="viridis", legend=False)
    plt.title('Mức độ đóng góp của các đặc trưng (Feature Importance)', fontsize=14, fontweight='bold')
    plt.xlabel('Mức độ quan trọng', fontsize=12)
    plt.ylabel('Đặc trưng', fontsize=12)
    plt.tight_layout()
    plt.savefig('3_feature_importance.png')
    plt.show()

    # ---------------------------------------------------------
    # 4. MULTI-CLASS ROC CURVE
    # ---------------------------------------------------------
    # Binarize nhãn để vẽ ROC cho bài toán đa lớp (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(['green', 'orange', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính giả (False Positive Rate)', fontsize=12)
    plt.ylabel('Tỷ lệ Dương tính thật (True Positive Rate)', fontsize=12)
    plt.title('Đường cong ROC Đa lớp (Multi-class ROC Curve)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('4_roc_curve.png')
    plt.show()


if __name__ == "__main__":
    filepath = 'simulation_survival_dataset4.csv'

    try:
        X, y, features = load_and_preprocess_data(filepath)

        # Biểu đồ 1: Phân bố dữ liệu
        plot_class_distribution(y)

        # Biểu đồ 2, 3, 4 và Báo cáo dạng text
        evaluate_model(X, y, features)

        print("\n✅ Đã lưu tất cả biểu đồ thành file ảnh (PNG) trong cùng thư mục!")

    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {filepath}. Hãy chắc chắn bạn đã chạy file gendata trước.")