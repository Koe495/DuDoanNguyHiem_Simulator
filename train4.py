import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_and_export_model():
    print("=== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH RANDOM FOREST ===")

    # 1. Đọc dữ liệu
    dataset_file = 'simulation_survival_dataset4.csv'
    try:
        df = pd.read_csv(dataset_file)
        print(f"✅ Đã tải dữ liệu thành công từ: {dataset_file}")
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file {dataset_file}!")
        print("Hãy chắc chắn bạn đã chạy file gendata4.py trước đó.")
        return

    # 2. Feature Engineering (Khớp tuyệt đối với logic của Game)
    print("Đang xử lý Feature Engineering (Trích xuất đặc trưng thời gian)...")
    # Sắp xếp theo người dùng và thời gian để tính toán chuỗi hợp lý
    df = df.sort_values(by=['user_id', 'timestamp'])

    # Tạo các cột đặc trưng mới
    df['hr_baseline'] = df.groupby('user_id')['hr'].transform('mean')
    df['hr_diff_from_baseline'] = df['hr'] - df['hr_baseline']
    df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(lambda x: x.rolling(window=12, min_periods=1).std())
    df['hr_rolling_std'] = df['hr_rolling_std'].fillna(0)  # Trám số 0 cho các giá trị NaN ở những giây đầu

    # 3. Chọn biến độc lập (X) và biến phụ thuộc (y)
    features = [
        'hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv',
        'body_temp', 'env_temp', 'humidity', 'pressure',
        'oxygen', 'uv_index', 'aqi', 'accel_mag', 'fall_duration'
    ]
    X = df[features]
    y = df['label']

    # 4. Huấn luyện mô hình
    print("Đang huấn luyện AI (100 Cây quyết định)...")
    # Sử dụng toàn bộ dữ liệu để train, class_weight='balanced' để xử lý lệch nhãn
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X, y)

    # 5. Xuất file mô hình
    model_filename = 'rf_model.pkl'
    joblib.dump(rf_model, model_filename)

    print("\n" + "=" * 50)
    print(f"✅ HOÀN THÀNH! Đã xuất file mô hình: {model_filename}")
    print("=" * 50)
    print("Bây giờ bạn có thể chạy file simulation_game.py. Game sẽ tự động nhận diện AI của bạn!")


if __name__ == "__main__":
    train_and_export_model()