import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


# ==========================================
# CÁC HÀM SINH DỮ LIỆU CHUỖI THỜI GIAN THEO KỊCH BẢN
# Mỗi kịch bản được sinh theo "Sự kiện" (Event) để đảm bảo tính liên tục của Time-series
# ==========================================

def generate_events(scenario_name, num_events, rows_per_event, label):
    """
    Hàm tổng quát để sinh các sự kiện.
    - num_events: Số lượng người dùng/sự kiện ảo.
    - rows_per_event: Số dòng dữ liệu (ticks) cho mỗi sự kiện.
    Tổng số dòng = num_events * rows_per_event
    """
    data = []

    for ev in range(num_events):
        user_id = f"User_{scenario_name}_{ev + 1:03d}"
        base_time = datetime.now() - timedelta(days=ev)  # Trải đều thời gian

        # Chỉ số cơ bản của người này (Baseline)
        hr = random.uniform(65.0, 80.0)
        hrv = random.uniform(50.0, 65.0)
        body_temp = random.uniform(36.5, 37.0)
        env_temp = 24.0
        humidity = 55.0
        pressure = 1013.0
        oxygen = 20.9
        uv = 2.0
        aqi = 35.0
        accel = 1.0
        fall_duration = 0

        for i in range(rows_per_event):
            current_time = base_time + timedelta(seconds=i * 5)  # Mỗi bước là 5 giây

            # ----------------------------------------------------
            # 1. NHÓM BÌNH THƯỜNG (LABEL 0)
            # ----------------------------------------------------
            if scenario_name == 'NORMAL':
                hr = np.clip(hr + random.uniform(-1, 1), 60, 95)
                hrv = np.clip(hrv + random.uniform(-0.5, 0.5), 45, 70)
                body_temp = np.clip(body_temp + random.uniform(-0.02, 0.02), 36.4, 37.3)
                env_temp = np.clip(env_temp + random.uniform(-0.1, 0.1), 20, 28)
                accel = np.clip(1.0 + random.uniform(-0.1, 0.1), 0.8, 1.2)  # Đang đi bộ bình thường

            # ----------------------------------------------------
            # 2. NHÓM DỰ ĐOÁN NGUY CƠ - CẢNH BÁO SỚM (LABEL 1)
            # ----------------------------------------------------
            elif scenario_name == 'WARN_INTERNAL_INJURY':  # Chấn thương nội ngầm
                if i == 0:
                    accel = 6.5  # Bị ngã mạnh ban đầu
                else:
                    accel = np.clip(1.0 + random.uniform(-0.05, 0.05), 0.9, 1.1)  # Lại đứng lên đi tiếp
                    hr += random.uniform(0.2, 0.8)  # Tim đập nhanh dần để bù máu
                    body_temp -= random.uniform(0.005, 0.015)  # Thân nhiệt giảm nhẹ do xuất huyết

            elif scenario_name == 'WARN_SILENT_POISON':  # Ngộ độc vô thức
                aqi = random.uniform(250, 400)  # Khí rất độc
                oxygen = np.clip(oxygen - random.uniform(0.01, 0.05), 17.0, 20.9)  # Oxy giảm dần
                hr += random.uniform(0.1, 0.5)  # Tim đập nhanh lên do thiếu oxy sinh học
                hrv -= random.uniform(0.2, 0.6)  # Suy giảm thần kinh

            elif scenario_name == 'WARN_ALTITUDE':  # Nguy cơ sốc độ cao
                pressure = np.clip(pressure - random.uniform(2.0, 5.0), 650, 1013)  # Áp suất tụt nhanh
                oxygen = np.clip(oxygen - random.uniform(0.05, 0.1), 15.0, 20.9)
                hr += random.uniform(0.3, 0.7)

            # ----------------------------------------------------
            # 3. NHÓM NGUY KỊCH TỬ VONG - SỰ CỐ THỰC SỰ (LABEL 2)
            # ----------------------------------------------------
            elif scenario_name == 'CRIT_SEVERE_FALL':  # Té ngã bất tỉnh
                if i == 0:
                    accel = 7.5  # Va chạm cực mạnh
                else:
                    accel = random.uniform(0.98, 1.02)  # Nằm im bất động
                    fall_duration = i * 5  # Thời gian ngã tăng liên tục
                    hr = np.clip(hr - random.uniform(0.5, 1.5), 40, 100)  # Nhịp tim lịm dần
                    body_temp -= random.uniform(0.02, 0.05)

            elif scenario_name == 'CRIT_HEATSTROKE':  # Sốc nhiệt cấp tính
                env_temp = random.uniform(41.0, 45.0)
                humidity = random.uniform(80.0, 95.0)
                body_temp = np.clip(body_temp + random.uniform(0.1, 0.3), 37.0, 41.5)  # Sốt cực cao
                hr = np.clip(hr + random.uniform(1.0, 3.0), 80, 175)  # Tim đập quá tải

            elif scenario_name == 'CRIT_HYPOTHERMIA':  # Hạ thân nhiệt cấp tính
                env_temp = random.uniform(-20.0, -10.0)
                body_temp = np.clip(body_temp - random.uniform(0.1, 0.2), 32.0, 37.0)  # Nhiệt độ cơ thể siêu thấp
                hr = np.clip(hr - random.uniform(0.5, 2.0), 35, 80)  # Tim gần ngừng đập

            # Đóng gói dữ liệu dòng hiện tại
            data.append([
                current_time, user_id,
                round(hr, 1), round(hrv, 1), round(body_temp, 2),
                round(env_temp, 1), round(humidity, 1), round(pressure, 1), round(oxygen, 2),
                round(uv, 1), int(aqi), round(accel, 2), fall_duration, label
            ])

    return data


# ==========================================
# THỰC THI KHỞI TẠO DATASET THEO YÊU CẦU
# ==========================================
if __name__ == "__main__":
    print("=== BẮT ĐẦU KHỞI TẠO DATASET SIMULATION 3 ===")

    all_data = []

    # 1. Sinh 10,000 data Bình Thường (Label 0)
    # Lấy 100 sự kiện (users), mỗi sự kiện kéo dài 100 dòng (500 giây)
    print("Đang sinh 10,000 dòng dữ liệu AN TOÀN (Label 0)...")
    all_data.extend(generate_events('NORMAL', num_events=100, rows_per_event=100, label=0))

    # 2. Sinh 3,000 data Cảnh Báo Nguy Cơ - 1,000 cho mỗi loại (Label 1)
    # Lấy 20 sự kiện, mỗi sự kiện 50 dòng
    print("Đang sinh 3,000 dòng dữ liệu CẢNH BÁO/DỰ ĐOÁN SỚM (Label 1)...")
    all_data.extend(generate_events('WARN_INTERNAL_INJURY', 20, 50, 1))
    all_data.extend(generate_events('WARN_SILENT_POISON', 20, 50, 1))
    all_data.extend(generate_events('WARN_ALTITUDE', 20, 50, 1))

    # 3. Sinh 1,500 data Nguy Kịch - 500 cho mỗi loại (Label 2)
    # Lấy 10 sự kiện, mỗi sự kiện 50 dòng
    print("Đang sinh 1,500 dòng dữ liệu NGUY KỊCH/TỬ VONG (Label 2)...")
    all_data.extend(generate_events('CRIT_SEVERE_FALL', 10, 50, 2))
    all_data.extend(generate_events('CRIT_HEATSTROKE', 10, 50, 2))
    all_data.extend(generate_events('CRIT_HYPOTHERMIA', 10, 50, 2))

    # Đưa toàn bộ vào DataFrame
    columns = [
        'timestamp', 'user_id', 'hr', 'hrv', 'body_temp', 'env_temp',
        'humidity', 'pressure', 'oxygen', 'uv_index', 'aqi',
        'accel_mag', 'fall_duration', 'label'
    ]
    df = pd.DataFrame(all_data, columns=columns)

    # Lưu ra file CSV
    output_filename = "simulation_survival_dataset3.csv"
    df.to_csv(output_filename, index=False)

    print("\n=== HOÀN THÀNH ===")
    print(f"Đã lưu dataset thành công vào file: '{output_filename}'")
    print(f"Tổng số dòng: {len(df)}")
    print("\nPhân bổ nhãn (Label Distribution):")
    print(df['label'].value_counts().sort_index())