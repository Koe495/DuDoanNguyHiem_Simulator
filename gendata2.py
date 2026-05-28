import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ==========================================
# CẤU HÌNH NGƯỜI DÙNG (USER PROFILES)
# ==========================================
# Việc tạo các profile khác nhau giúp AI học được cách "Cá nhân hóa baseline"
USER_PROFILES = {
    'User_01_Athlete': {'hr': 60.0, 'hrv': 70.0, 'temp': 36.5},  # VĐV chuyên nghiệp: Tim đập chậm, HRV cao
    'User_02_Normal': {'hr': 75.0, 'hrv': 50.0, 'temp': 36.8},  # Người bình thường
    'User_03_Older': {'hr': 85.0, 'hrv': 35.0, 'temp': 36.9},  # Người lớn tuổi: Tim đập nhanh hơn, HRV thấp
}


def generate_user_data(user_id, profile, num_rows=5000, start_time=datetime.now()):
    """Sinh dữ liệu bình thường cho MỘT người dùng cụ thể"""
    timestamps = [start_time + timedelta(seconds=5 * i) for i in range(num_rows)]

    data = {
        'timestamp': timestamps,
        'user_id': [user_id] * num_rows,
        'hr': np.zeros(num_rows), 'hrv': np.zeros(num_rows), 'body_temp': np.zeros(num_rows),
        'accel_mag': np.zeros(num_rows), 'fall_duration': np.zeros(num_rows),
        'env_temp': np.zeros(num_rows), 'humidity': np.zeros(num_rows), 'pressure': np.zeros(num_rows),
        'oxygen': np.zeros(num_rows), 'uv_index': np.zeros(num_rows), 'aqi': np.zeros(num_rows),
        'is_anomaly': [0] * num_rows
    }

    # Lấy baseline từ profile của người dùng
    hr, hrv, body_temp = profile['hr'], profile['hrv'], profile['temp']
    env_temp, humidity, pressure, oxygen, uv, aqi = 24.0, 55.0, 1013.0, 20.9, 2.0, 35.0

    # Random Walk
    for i in range(num_rows):
        hr = np.clip(hr + np.random.normal(0, 0.5), profile['hr'] - 10, profile['hr'] + 15)
        hrv = np.clip(hrv + np.random.normal(0, 0.4), profile['hrv'] - 15, profile['hrv'] + 15)
        body_temp = np.clip(body_temp + np.random.normal(0, 0.01), 36.5, 37.2)
        accel_mag = np.clip(1.0 + np.random.normal(0, 0.1), 0.8, 1.4)  # Đi bộ bình thường ~ 1G

        env_temp = np.clip(env_temp + np.random.normal(0, 0.05), 15, 30)
        humidity = np.clip(humidity + np.random.normal(0, 0.2), 40, 70)
        pressure = np.clip(pressure + np.random.normal(0, 0.1), 1000, 1020)
        oxygen = np.clip(oxygen + np.random.normal(0, 0.002), 20.7, 20.9)
        uv = np.clip(uv + np.random.normal(0, 0.01), 1, 5)
        aqi = int(np.clip(aqi + np.random.normal(0, 0.5), 20, 60))

        data['hr'][i] = round(hr, 1)
        data['hrv'][i] = round(hrv, 1)
        data['body_temp'][i] = round(body_temp, 2)
        data['accel_mag'][i] = round(accel_mag, 2)
        data['env_temp'][i] = round(env_temp, 1)
        data['humidity'][i] = round(humidity, 1)
        data['pressure'][i] = round(pressure, 1)
        data['oxygen'][i] = round(oxygen, 2)
        data['uv_index'][i] = round(uv, 1)
        data['aqi'][i] = aqi

    return pd.DataFrame(data)


def inject_anomalies_for_user(df, user_id):
    """Chèn ngẫu nhiên 2 biến cố nguy hiểm vào timeline của người dùng này"""
    # Danh sách 5 kịch bản nguy hiểm có thể xảy ra
    all_events = ['heatstroke', 'hypoxia', 'severe_fall', 'hypothermia', 'toxic_gas']

    # Chọn ngẫu nhiên 2 biến cố để người dùng này gặp phải
    chosen_events = random.sample(all_events, 2)
    print(f" -> {user_id} sẽ gặp các biến cố: {chosen_events}")

    for event in chosen_events:
        # Chọn vị trí bắt đầu ngẫu nhiên (tránh chèn vào cuối file)
        start_idx = random.randint(500, len(df) - 100)

        if event == 'heatstroke':  # 1. SỐC NHIỆT (Môi trường nóng, tim đập nhanh)
            length = 150
            for idx in range(start_idx, start_idx + length):
                df.loc[idx, 'env_temp'] = round(random.uniform(41.0, 45.0), 1)
                df.loc[idx, 'hr'] = round(random.uniform(150.0, 175.0), 1)
                df.loc[idx, 'body_temp'] = round(random.uniform(39.5, 41.0), 2)
                df.loc[idx, 'hrv'] = round(random.uniform(10.0, 15.0), 1)
                df.loc[idx, 'is_anomaly'] = 1

        elif event == 'hypoxia':  # 2. THIẾU OXY ĐỘ CAO (Áp suất thấp, Oxy giảm)
            length = 200
            for idx in range(start_idx, start_idx + length):
                df.loc[idx, 'pressure'] = round(random.uniform(600.0, 650.0), 1)
                df.loc[idx, 'oxygen'] = round(random.uniform(12.0, 14.5), 2)
                df.loc[idx, 'hr'] = round(random.uniform(130.0, 150.0), 1)
                df.loc[idx, 'is_anomaly'] = 1

        elif event == 'severe_fall':  # 3. TÉ NGÃ (Gia tốc vọt lên rồi nằm im)
            length = 250
            df.loc[start_idx, 'accel_mag'] = round(random.uniform(5.0, 7.0), 2)  # Cú va chạm
            df.loc[start_idx, 'is_anomaly'] = 1
            for i, idx in enumerate(range(start_idx + 1, start_idx + length)):
                df.loc[idx, 'accel_mag'] = round(random.uniform(0.95, 1.05), 2)  # Nằm im
                df.loc[idx, 'fall_duration'] = (i + 1) * 5
                df.loc[idx, 'hr'] = round(random.uniform(45.0, 55.0), 1)  # Tim lịm đi
                df.loc[idx, 'is_anomaly'] = 1

        elif event == 'hypothermia':  # 4. HẠ THÂN NHIỆT (Bão tuyết / Rơi xuống nước đá)
            length = 20
            for idx in range(start_idx, start_idx + length):
                df.loc[idx, 'env_temp'] = round(random.uniform(-15.0, -5.0), 1)
                df.loc[idx, 'body_temp'] = round(random.uniform(33.0, 35.0), 2)  # Thân nhiệt nguy kịch
                df.loc[idx, 'hr'] = round(random.uniform(40.0, 50.0), 1)  # Mạch đập cực chậm
                df.loc[idx, 'hrv'] = round(random.uniform(10.0, 12.0), 1)
                df.loc[idx, 'is_anomaly'] = 1

        elif event == 'toxic_gas':  # 5. NGỘ ĐỘC KHÍ / KẸT TRONG HANG (AQI cực cao)
            length = 150
            for idx in range(start_idx, start_idx + length):
                df.loc[idx, 'aqi'] = int(random.uniform(400, 600))  # Không khí vô cùng độc
                df.loc[idx, 'oxygen'] = round(random.uniform(17.0, 18.5), 2)
                df.loc[idx, 'hr'] = round(random.uniform(120.0, 140.0), 1)  # Hoảng loạn, khó thở
                df.loc[idx, 'is_anomaly'] = 1

    return df


if __name__ == "__main__":
    print("=== BẮT ĐẦU MÔ PHỎNG DỮ LIỆU ĐA NGƯỜI DÙNG ===")

    all_user_data = []

    # Duyệt qua từng User để sinh dữ liệu
    for user_id, profile in USER_PROFILES.items():
        print(f"\nĐang xử lý {user_id}...")

        # 1. Sinh 5000 dòng dữ liệu bình thường cho user này
        df_normal = generate_user_data(user_id, profile, num_rows=5000)

        # 2. Chèn biến cố nguy hiểm riêng cho user này
        df_injected = inject_anomalies_for_user(df_normal, user_id)

        # Đưa vào mảng tổng
        all_user_data.append(df_injected)

    # Nối dữ liệu của tất cả người dùng lại thành 1 Dataframe duy nhất
    final_dataset = pd.concat(all_user_data, ignore_index=True)

    # Lưu ra file CSV
    output_filename = "multi_user_survival_dataset2.csv"
    final_dataset.to_csv(output_filename, index=False)

    print(f"\n=== HOÀN THÀNH ===")
    print(f"Đã lưu dataset thành công vào file: '{output_filename}'")
    print(f"Tổng số bản ghi: {len(final_dataset)} dòng.")
    print(f"Tổng số dòng chứa kịch bản NGUY HIỂM: {final_dataset['is_anomaly'].sum()} dòng.")