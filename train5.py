import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("🔄 Đang nạp dữ liệu từ 'simulation_survival_dataset5.csv'...")

# ==========================================
# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
try:
    df = pd.read_csv('simulation_survival_dataset5.csv')
    print(f"✅ Đã tải thành công {len(df)} dòng dữ liệu.")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'simulation_survival_dataset5.csv'. Vui lòng kiểm tra lại thư mục.")
    exit()

# Tạo các đặc trưng phái sinh (Feature Engineering) để khớp với game
df = df.sort_values(by=['user_id', 'timestamp'])

# Tính nhịp tim nền tảng động dựa trên trung bình thực tế của từng user
df['hr_baseline'] = df.groupby('user_id')['hr'].transform('mean')
df['hr_diff_from_baseline'] = df['hr'] - df['hr_baseline']

# Tính độ lệch chuẩn nhịp tim trượt (Rolling Standard Deviation) với cửa sổ window=12 đồng bộ
df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(
    lambda x: x.rolling(window=12, min_periods=1).std()
)
df['hr_rolling_std'] = df['hr_rolling_std'].fillna(0)
# ==========================================
# 2. CHUẨN BỊ FEATURES (X) VÀ TARGET (y)
# ==========================================
# Danh sách 13 đặc trưng bắt buộc khớp với cấu trúc trong game
features = [
    'hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv', 'body_temp',
    'env_temp', 'humidity', 'pressure', 'oxygen', 'uv_index', 'aqi',
    'accel_mag', 'fall_duration'
]

X = df[features]
y = df['label']  # Cột nhãn dự đoán (0: An toàn, 1: Cảnh báo, 2: Nguy kịch)

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm thử (20%)
# stratify=y đảm bảo tỷ lệ nhãn phân bổ đều giữa tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. HUẤN LUYỆN MÔ HÌNH HỌC MÁY
# ==========================================
print("\n🧠 Đang huấn luyện Random Forest Classifier...")
# Sử dụng 150 cây quyết định, giới hạn độ sâu 15 để tránh overfitting
model = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ==========================================
# 4. ĐÁNH GIÁ VÀ XUẤT MÔ HÌNH
# ==========================================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 ĐỘ CHÍNH XÁC TRÊN TẬP KIỂM THỬ: {accuracy * 100:.2f}%")
print("📊 Báo cáo phân loại chi tiết:")
print(classification_report(y_test, y_pred, target_names=['Safe (0)', 'Caution (1)', 'Critical (2)']))

# Lưu model với tên mới để quản lý phiên bản
model_filename = 'rf_model_v5.pkl'
joblib.dump(model, model_filename)
print(f"\n✅ Đã lưu mô hình thành công vào '{model_filename}'!")
print("🚀 Đừng quên cập nhật đường dẫn load model trong file game của bạn nhé.")