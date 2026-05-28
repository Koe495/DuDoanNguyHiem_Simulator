import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=== BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU ===")
# 1. Đọc dataset đã tạo từ Simulator
df = pd.read_csv('multi_user_survival_dataset3.csv')
print(f"Đã tải {len(df)} dòng dữ liệu.")

print("\n=== BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING) ===")

# Kỹ thuật 1: Baseline Normalization (Độ lệch so với Baseline cá nhân)
# Tính nhịp tim trung bình của TỪNG người dùng, sau đó xem hiện tại họ đang lệch bao nhiêu
df['hr_baseline'] = df.groupby('user_id')['hr'].transform('mean')
df['hr_diff_from_baseline'] = df['hr'] - df['hr_baseline']

# Kỹ thuật 2: Sliding Window (Cửa sổ thời gian)
# Tính phương sai/độ lệch chuẩn của nhịp tim trong 1 phút qua (12 dòng x 5 giây = 60s)
# Giúp phát hiện sự "mất ổn định" đột ngột của nhịp tim
df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(lambda x: x.rolling(window=12, min_periods=1).std())
df['hr_rolling_std'].fillna(0, inplace=True) # Điền 0 cho những giây đầu tiên chưa đủ cửa sổ

print("Đã trích xuất xong các đặc trưng: 'hr_diff_from_baseline' và 'hr_rolling_std'.")

print("\n=== BƯỚC 3: CHUẨN BỊ TẬP HUẤN LUYỆN VÀ KIỂM THỬ ===")
# Chọn các cột đặc trưng (Features - X) đưa vào mô hình
features = [
    'hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv',
    'body_temp', 'env_temp', 'humidity', 'pressure',
    'oxygen', 'uv_index', 'aqi', 'accel_mag', 'fall_duration'
]
X = df[features]

# Chọn cột nhãn (Target - y)
y = df['is_anomaly']

# Chia tỷ lệ 80% để học (Train), 20% để thi (Test).
# stratify=y đảm bảo tỷ lệ dữ liệu nguy hiểm được chia đều ở cả 2 tập.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Dữ liệu huấn luyện (Train): {len(X_train)} dòng.")
print(f"Dữ liệu kiểm thử (Test): {len(X_test)} dòng.")

print("\n=== BƯỚC 4: HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ===")
# Khởi tạo mô hình với 100 cây quyết định (n_estimators=100)
# class_weight='balanced': Cực kỳ quan trọng! Giúp AI chú ý nhiều hơn vào nhóm "Nguy hiểm" (vốn rất ít dữ liệu)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Cho AI học từ dữ liệu
rf_model.fit(X_train, y_train)
print("Huấn luyện mô hình thành công!")

print("\n=== BƯỚC 5: ĐÁNH GIÁ HIỆU SUẤT (PERFORMANCE - P) ===")
# Cho AI làm bài thi trên tập Test
y_pred = rf_model.predict(X_test)

# In báo cáo kết quả
print("\n--- BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ---")
print(classification_report(y_test, y_pred, target_names=['0: An toàn', '1: Nguy hiểm']))

# Vẽ Ma trận nhầm lẫn (Confusion Matrix) để xem AI có đoán sai nhiều không
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Đoán An toàn', 'Đoán Nguy hiểm'], yticklabels=['Thực tế An toàn', 'Thực tế Nguy hiểm'])
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.ylabel('Thực tế (Ground Truth)')
plt.xlabel('AI Dự đoán (Predictions)')
plt.show()

print("\n=== BƯỚC 6: XUẤT ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE) ===")
# Xem AI dựa vào yếu tố nào nhiều nhất để quyết định là "Nguy hiểm"
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_imp_df)