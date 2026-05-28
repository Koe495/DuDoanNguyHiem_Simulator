import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("🔄 Đang nạp dữ liệu từ 'simulation_survival_dataset4.csv'...")

# ==========================================
# 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
try:
    df = pd.read_csv('simulation_survival_dataset4.csv')
    print(f"✅ Đã tải thành công {len(df)} dòng dữ liệu.")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'simulation_survival_dataset4.csv'. Vui lòng kiểm tra lại thư mục.")
    exit()

# Tạo các đặc trưng phái sinh (Feature Engineering) để khớp với game
# Giả định nhịp tim cơ bản trung bình của các user là 75 bpm để tính độ lệch
df['hr_diff_from_baseline'] = df['hr'] - 75.0

# Tính độ lệch chuẩn nhịp tim (Rolling Standard Deviation) theo từng user để đo lường biến thiên
df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(
    lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
)

# ==========================================
# 2. CHUẨN BỊ FEATURES (X) VÀ TARGET (y)
# ==========================================
# Danh sách này KHỚP 100% với mảng columns trong game
features = [
    'hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv', 'body_temp',
    'env_temp', 'humidity', 'pressure', 'oxygen', 'uv_index', 'aqi',
    'accel_mag', 'fall_duration'
]

X = df[features]
y = df['label']  # Cột nhãn dự đoán (0: Bình thường, 1: Cảnh báo, 2: Nguy kịch)

# ĐÃ SỬA LỖI TẠI ĐÂY: Dùng test_size thay vì test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. HUẤN LUYỆN MÔ HÌNH HỌC MÁY
# ==========================================
print("\n🧠 Đang huấn luyện Random Forest Classifier...")
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

# Lưu model
joblib.dump(model, 'rf_model2.pkl')
print("\n✅ Đã lưu mô hình thành công vào 'rf_model.pkl'!")
print("🚀 Bạn có thể mở game lên trải nghiệm ngay bây giờ.")