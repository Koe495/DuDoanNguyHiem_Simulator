import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_normal_data(num_rows=10000):
    print(f"--- Bước 1: Đang khởi tạo {num_rows} dòng dữ liệu 'Bình thường' ---")

    # Khởi tạo thời gian bắt đầu
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=5 * i) for i in range(num_rows)]

    # Thiết lập giá trị ban đầu (Baseline) và biên độ dao động an toàn
    data = {
        'timestamp': timestamps,
        'user_id': ['User_01'] * num_rows,

        # Dữ liệu sinh lý
        'hr': np.zeros(num_rows),  # Nhịp tim (bpm)
        'hrv': np.zeros(num_rows),  # Biến thiên nhịp tim (ms)
        'body_temp': np.zeros(num_rows),  # Nhiệt độ cơ thể (°C)
        'accel_mag': np.zeros(num_rows),  # Độ lớn gia tốc kế (g)
        'fall_duration': np.zeros(num_rows),  # Thời gian nằm im sau ngã (giây)

        # Dữ liệu môi trường
        'env_temp': np.zeros(num_rows),  # Nhiệt độ môi trường (°C)
        'humidity': np.zeros(num_rows),  # Độ ẩm (%)
        'pressure': np.zeros(num_rows),  # Áp suất khí quyển (hPa)
        'oxygen': np.zeros(num_rows),  # Nồng độ Oxy (%)
        'uv_index': np.zeros(num_rows),  # Chỉ số UV
        'aqi': np.zeros(num_rows),  # Chất lượng không khí

        # Nhãn mục tiêu cho bài toán Phát hiện bất thường
        'is_anomaly': [0] * num_rows  # 0: Bình thường, 1: Bất thường/Nguy hiểm
    }

    # Cài đặt giá trị xuất phát ban đầu
    hr = 80.0
    hrv = 55.0
    body_temp = 36.8
    env_temp = 24.0
    humidity = 55.0
    pressure = 1013.0
    oxygen = 20.9
    uv = 2.0
    aqi = 35.0

    # Sử dụng Random Walk để dữ liệu chuỗi thời gian biến thiên tự nhiên
    for i in range(num_rows):
        hr = np.clip(hr + np.random.normal(0, 0.5), 65, 100)
        hrv = np.clip(hrv + np.random.normal(0, 0.4), 40, 70)
        body_temp = np.clip(body_temp + np.random.normal(0, 0.01), 36.5, 37.2)

        # Người thám hiểm di chuyển bình thường, gia tốc dao động quanh mức 1G, không ngã
        accel_mag = np.clip(1.0 + np.random.normal(0, 0.1), 0.8, 1.4)

        env_temp = np.clip(env_temp + np.random.normal(0, 0.02), 18, 28)
        humidity = np.clip(humidity + np.random.normal(0, 0.1), 45, 65)
        pressure = np.clip(pressure + np.random.normal(0, 0.05), 1005, 1018)
        oxygen = np.clip(oxygen + np.random.normal(0, 0.005), 20.7, 20.9)
        uv = np.clip(uv + np.random.normal(0, 0.01), 1, 4)
        aqi = int(np.clip(aqi + np.random.normal(0, 0.2), 20, 50))

        # Gán vào mảng dữ liệu
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


def inject_anomalies(df, total_anomaly_rows=50):
    print(f"\n--- Bước 2: Tiến hành chèn khoảng {total_anomaly_rows} dòng dữ liệu 'Nguy hiểm' ---")

    # Để phù hợp với chuỗi thời gian, chúng ta chia 50 dòng thành 3 "Biến cố" nguy hiểm kéo dài liên tục
    # Biến cố 1: Sốc nhiệt (Heatstroke) - Khoảng 15 dòng
    # Biến cố 2: Thiếu Oxy do độ cao (Hypoxia) - Khoảng 15 dòng
    # Biến cố 3: Té ngã nghiêm trọng bất tỉnh (Severe Fall) - Khoảng 20 dòng

    # --- BIẾN CỐ 1: SỐC NHIỆT (Vị trí ngẫu nhiên trong khoảng 2000 - 3000) ---
    start_idx_1 = random.randint(2000, 2500)
    len_1 = 15
    print(f"[Biến cố 1] Chèn kịch bản SỐC NHIỆT tại dòng {start_idx_1} đến {start_idx_1 + len_1}")
    for idx in range(start_idx_1, start_idx_1 + len_1):
        df.loc[idx, 'env_temp'] = round(random.uniform(41.0, 44.0), 1)  # Môi trường quá nóng
        df.loc[idx, 'humidity'] = round(random.uniform(75.0, 85.0), 1)  # Độ ẩm ngột ngạt
        df.loc[idx, 'hr'] = round(random.uniform(145.0, 165.0), 1)  # Nhịp tim tăng vọt cực hạn
        df.loc[idx, 'hrv'] = round(random.uniform(10.0, 18.0), 1)  # HRV sụt giảm mạnh (stress nặng)
        df.loc[idx, 'body_temp'] = round(random.uniform(39.2, 40.5), 2)  # Sốt cao do sốc nhiệt
        df.loc[idx, 'is_anomaly'] = 1

    # --- BIẾN CỐ 2: THIẾU OXY DO ĐỘ CAO (Vị trí ngẫu nhiên trong khoảng 5000 - 6000) ---
    start_idx_2 = random.randint(5000, 5500)
    len_2 = 15
    print(f"[Biến cố 2] Chèn kịch bản THIẾU OXY độ cao tại dòng {start_idx_2} đến {start_idx_2 + len_2}")
    for idx in range(start_idx_2, start_idx_2 + len_2):
        df.loc[idx, 'pressure'] = round(random.uniform(650.0, 720.0), 1)  # Áp suất giảm sâu (lên núi cực cao)
        df.loc[idx, 'oxygen'] = round(random.uniform(13.0, 15.5), 2)  # Oxy loãng đe dọa tính mạng
        df.loc[idx, 'hr'] = round(random.uniform(125.0, 140.0), 1)  # Tim đập nhanh để bù oxy
        df.loc[idx, 'hrv'] = round(random.uniform(15.0, 22.0), 1)
        df.loc[idx, 'env_temp'] = round(random.uniform(-5.0, 2.0), 1)  # Trên núi cao thường rất lạnh
        df.loc[idx, 'is_anomaly'] = 1

    # --- BIẾN CỐ 3: TÉ NGÃ NGHIÊM TRỌNG BẤT TỈNH (Vị trí ngẫu nhiên trong khoảng 8000 - 9000) ---
    start_idx_3 = random.randint(8000, 8500)
    len_3 = 20
    print(f"[Biến cố 3] Chèn kịch bản TÉ NGÃ CHẤN THƯƠNG tại dòng {start_idx_3} đến {start_idx_3 + len_3}")

    # Dòng đầu tiên của biến cố là cú va chạm mạnh (Gia tốc cực lớn)
    df.loc[start_idx_3, 'accel_mag'] = round(random.uniform(4.5, 6.0), 2)  # Cú sốc va chạm va đập mạnh > 4G
    df.loc[start_idx_3, 'is_anomaly'] = 1

    # Các dòng tiếp theo người thám hiểm nằm bất động, thời gian ngã tăng dần
    for i, idx in enumerate(range(start_idx_3 + 1, start_idx_3 + len_3)):
        df.loc[idx, 'accel_mag'] = round(random.uniform(0.98, 1.02), 2)  # Nằm im bất động (chỉ chịu trọng lực ~1G)
        df.loc[idx, 'fall_duration'] = (i + 1) * 5  # Tăng thêm 5 giây sau mỗi dòng dữ liệu
        df.loc[idx, 'hr'] = round(random.uniform(50.0, 60.0), 1)  # Nhịp tim suy giảm nguy kịch do bất tỉnh
        df.loc[idx, 'hrv'] = round(random.uniform(12.0, 20.0), 1)
        df.loc[idx, 'body_temp'] = round(df.loc[idx - 1, 'body_temp'] - 0.1, 2)  # Thân nhiệt giảm dần do mất máu/nằm im
        df.loc[idx, 'is_anomaly'] = 1

    return df


if __name__ == "__main__":
    # Chạy quy trình tạo Dataset mô phỏng
    df_normal = generate_normal_data(num_rows=10000)
    df_final = inject_anomalies(df_normal, total_anomaly_rows=50)

    # Lưu kết quả ra file CSV để làm tài nguyên huấn luyện Học máy
    output_filename = "simulation_survival_dataset.csv"
    df_final.to_csv(output_filename, index=False)

    print(f"\n Hoàn thành! Dataset đã được lưu thành công vào file: '{output_filename}'")
    print(f"Tổng số bản ghi: {len(df_final)}")
    print(f"Số dòng dữ liệu nguy hiểm (bất thường): {df_final['is_anomaly'].sum()}")

    # Hiển thị thử một vài dòng dữ liệu nguy hiểm để kiểm tra
    print("\n--- Bản xem trước một số dòng dữ liệu Sốc nhiệt được chèn vào: ---")
    print(df_final[df_final['is_anomaly'] == 1].head(5)[
              ['timestamp', 'hr', 'body_temp', 'env_temp', 'oxygen', 'is_anomaly']])