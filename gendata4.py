import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_events(scenario_name, num_events, rows_per_event, label):
    data = []

    for ev in range(num_events):
        user_id = f"User_{scenario_name}_{ev + 1:03d}"
        base_time = datetime.now() - timedelta(days=ev)

        # Baseline tự nhiên của mỗi người
        hr = random.uniform(65.0, 80.0)
        hrv = random.uniform(50.0, 65.0)
        body_temp = random.uniform(36.5, 37.0)
        env_temp = random.uniform(20.0, 26.0)
        humidity = random.uniform(50.0, 60.0)
        pressure = random.uniform(1000.0, 1015.0)
        oxygen = 20.9
        uv = random.uniform(1.0, 3.0)
        aqi = random.uniform(20.0, 50.0)
        accel = 1.0
        fall_duration = 0

        for i in range(rows_per_event):
            current_time = base_time + timedelta(seconds=i * 5)

            # ==========================================
            # NHÓM 0: AN TOÀN (Bao gồm cả Nhiễu)
            # ==========================================
            if scenario_name == 'NORMAL_WALKING':
                hr = np.clip(hr + random.uniform(-2, 2), 60, 100)
                body_temp = np.clip(body_temp + random.uniform(-0.02, 0.02), 36.4, 37.3)
                accel = np.clip(1.0 + random.uniform(-0.2, 0.2), 0.8, 1.3)

            elif scenario_name == 'NORMAL_HEAVY_CLIMB':  # NHIỄU: Nhịp tim và thân nhiệt rất cao do ráng sức
                hr = np.clip(hr + random.uniform(1, 3), 130, 165)  # Dễ nhầm với Sốc nhiệt
                body_temp = np.clip(body_temp + random.uniform(0.01, 0.05), 37.0, 37.8)  # Nóng do vận động
                hrv = np.clip(hrv - random.uniform(0.5, 1.0), 30, 45)  # Mệt mỏi tự nhiên
                accel = np.clip(1.0 + random.uniform(-0.3, 0.5), 0.9, 1.8)

            elif scenario_name == 'NORMAL_JUMPING':  # NHIỄU: Gia tốc vọt cao nhưng không ngã
                if i % 10 == 0:  # Thỉnh thoảng nhảy một cú mạnh
                    accel = random.uniform(3.5, 5.5)  # Dễ nhầm với ngã chấn thương
                else:
                    accel = np.clip(1.0 + random.uniform(-0.1, 0.1), 0.9, 1.2)
                hr = np.clip(hr + random.uniform(-1, 2), 80, 110)

            # ==========================================
            # NHÓM 1: CẢNH BÁO SỚM (Bao gồm cả Nhiễu)
            # ==========================================
            elif scenario_name == 'WARN_INTERNAL_INJURY':
                if i == 0:
                    accel = random.uniform(4.5, 6.5)  # Ngã mạnh
                else:
                    accel = np.clip(1.0 + random.uniform(-0.05, 0.05), 0.9, 1.1)
                hr += random.uniform(0.2, 0.8)  # Tăng dần để bù máu
                body_temp -= random.uniform(0.005, 0.015)

            elif scenario_name == 'WARN_SILENT_POISON':
                aqi = random.uniform(150, 350)  # Mức độ độc dao động, không cố định
                oxygen = np.clip(oxygen - random.uniform(0.01, 0.05), 17.5, 20.9)
                hr += random.uniform(0.1, 0.6)
                hrv -= random.uniform(0.2, 0.6)

            elif scenario_name == 'WARN_SLEEP_HYPOTHERMIA':  # NHIỄU: Nguy hiểm ngầm trong lúc ngủ
                hr = np.clip(hr - random.uniform(0.1, 0.5), 45, 60)  # Nhịp tim thấp giống như đang ngủ bình thường
                env_temp = random.uniform(-5.0, 5.0)  # Trời lạnh
                body_temp -= random.uniform(0.01, 0.03)  # Thân nhiệt tụt ngầm xuống mức 35.5 - Dễ bị bỏ qua

            # ==========================================
            # NHÓM 2: NGUY KỊCH
            # ==========================================
            elif scenario_name == 'CRIT_SEVERE_FALL':
                if i == 0:
                    accel = random.uniform(6.0, 8.5)
                else:
                    accel = random.uniform(0.95, 1.05)
                    fall_duration = i * 5
                    hr = np.clip(hr - random.uniform(0.5, 1.5), 40, 110)
                    body_temp -= random.uniform(0.02, 0.05)

            elif scenario_name == 'CRIT_HEATSTROKE':
                env_temp = random.uniform(38.0, 45.0)
                body_temp = np.clip(body_temp + random.uniform(0.1, 0.2), 38.5, 41.0)
                hr = np.clip(hr + random.uniform(1.0, 2.5), 140, 180)  # Ranh giới chồng lấn với NORMAL_HEAVY_CLIMB

            # Làm tròn số để giống cảm biến thực
            data.append([
                current_time, user_id,
                round(hr, 1), round(hrv, 1), round(body_temp, 2),
                round(env_temp, 1), round(humidity, 1), round(pressure, 1), round(oxygen, 2),
                round(uv, 1), int(aqi), round(accel, 2), fall_duration, label
            ])

    return data


if __name__ == "__main__":
    print("=== BẮT ĐẦU KHỞI TẠO DATASET SIMULATION 4 (CÓ NHIỄU) ===")

    all_data = []

    # 1. NHÓM 0: AN TOÀN (6,500 dòng chuẩn + 3,500 dòng nhiễu)
    print("Đang sinh dữ liệu Label 0 (Bao gồm đi bộ, leo núi nặng, nhảy Parkour)...")
    all_data.extend(generate_events('NORMAL_WALKING', 65, 100, 0))
    all_data.extend(generate_events('NORMAL_HEAVY_CLIMB', 20, 100, 0))  # Nhiễu nhịp tim cao
    all_data.extend(generate_events('NORMAL_JUMPING', 15, 100, 0))  # Nhiễu gia tốc mạnh

    # 2. NHÓM 1: CẢNH BÁO (3,000 dòng bao gồm kịch bản nhiễu khi ngủ)
    print("Đang sinh dữ liệu Label 1 (Dự đoán rủi ro ngầm)...")
    all_data.extend(generate_events('WARN_INTERNAL_INJURY', 20, 50, 1))
    all_data.extend(generate_events('WARN_SILENT_POISON', 20, 50, 1))
    all_data.extend(generate_events('WARN_SLEEP_HYPOTHERMIA', 20, 50, 1))  # Nhiễu nhịp tim thấp

    # 3. NHÓM 2: NGUY KỊCH (1,500 dòng)
    print("Đang sinh dữ liệu Label 2 (Sự cố cấp tính)...")
    all_data.extend(generate_events('CRIT_SEVERE_FALL', 10, 50, 2))
    all_data.extend(generate_events('CRIT_HEATSTROKE', 20, 50, 2))  # Tăng số lượng để test chồng lấn

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'user_id', 'hr', 'hrv', 'body_temp', 'env_temp',
        'humidity', 'pressure', 'oxygen', 'uv_index', 'aqi',
        'accel_mag', 'fall_duration', 'label'
    ])

    # Xáo trộn nhẹ để tăng tính ngẫu nhiên của mô hình huấn luyện
    # Nhưng vẫn giữ thứ tự thời gian cho từng user bằng cách sort lại sau khi load
    output_filename = "simulation_survival_dataset4.csv"
    df.to_csv(output_filename, index=False)

    print(f"\n✅ Đã lưu thành công: {output_filename}")
    print(df['label'].value_counts().sort_index())