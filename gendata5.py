import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def simulate_user_timeline(scenario_name, num_events, rows_per_event, default_label):
    """
    Mô phỏng chuỗi thời gian sinh lý dựa trên cơ chế phản hồi sinh học liên tục,
    loại bỏ hoàn toàn việc gán cứng các dải số cảm biến độc lập.
    """
    data = []

    for ev in range(num_events):
        user_id = f"User_{scenario_name}_{ev + 1:03d}"
        base_time = datetime.now() - timedelta(days=ev)

        # 1. KHỞI TẠO NỀN TẢNG SINH LÝ TỰ NHIÊN CỦA MỖI NGƯỜI (Khác biệt cá thể)
        base_hr = random.uniform(65.0, 75.0)
        base_hrv = random.uniform(55.0, 70.0)
        base_temp = 36.8

        # Các biến trạng thái cơ thể hiện tại
        hr = base_hr
        hrv = base_hrv
        body_temp = base_temp

        # 2. ĐỊNH NGHĨA CÁC ĐỘNG LỰC HỌC SƠ CẤP THEO KỊCH BẢN
        exertion = 0.1  # Mức độ gắng sức thể chất (0.0: nghỉ ngơi, 1.0: tối đa)
        clothing = 0.5  # Độ ấm áo mặc (0.0: cởi trần, 1.0: áo khoác dày bọc kín)
        trauma_shock = 0.0  # Mức độ sốc chấn thương do tai nạn (0.0 -> 1.0)
        oxygen_env = 20.9  # Oxy môi trường (%)
        aqi_env = 25.0  # Chỉ số ô nhiễm không khí
        env_temp = 24.0  # Nhiệt độ môi trường (°C)
        humidity = random.uniform(50.0, 65.0)
        pressure = random.uniform(1005.0, 1013.0)
        uv_index = random.uniform(1.0, 3.0)

        # Thiết lập cấu hình ban đầu cho từng kịch bản
        if scenario_name == 'NORMAL_WALKING':
            exertion = random.uniform(0.15, 0.3)
            env_temp = random.uniform(18.0, 26.0)
        elif scenario_name == 'NORMAL_HEAVY_CLIMB':
            exertion = random.uniform(0.75, 0.9)  # Gắng sức cực cao
            env_temp = random.uniform(22.0, 28.0)
        elif scenario_name == 'NORMAL_JUMPING':
            exertion = 0.4
            env_temp = random.uniform(20.0, 25.0)
        elif scenario_name == 'WARN_INTERNAL_INJURY':
            exertion = 0.05  # Đau đớn, hạn chế vận động
            env_temp = random.uniform(15.0, 22.0)
        elif scenario_name == 'WARN_SILENT_POISON':
            exertion = 0.1
            aqi_env = random.uniform(180.0, 280.0)  # Môi trường nhiễm độc nhẹ đến vừa
            oxygen_env = random.uniform(17.0, 18.5)
        elif scenario_name == 'WARN_SLEEP_HYPOTHERMIA':
            exertion = -0.2  # Suy giảm chuyển hóa khi ngủ sâu
            env_temp = random.uniform(-10.0, 2.0)  # Trời lạnh giá
            clothing = 0.25  # Mặc quần áo quá mỏng
        elif scenario_name == 'CRIT_SEVERE_FALL':
            exertion = -0.3  # Ngất xỉu, hôn mê sâu
            env_temp = random.uniform(10.0, 18.0)
        elif scenario_name == 'CRIT_HEATSTROKE':
            exertion = 0.7  # Vừa gắng sức nặng vừa nắng nóng
            env_temp = random.uniform(39.0, 45.0)  # Sa mạc cực đoan
            clothing = 0.8  # Mặc đồ bảo hộ dày giữ nhiệt

        # 3. VÒNG LẶP CHUỖI THỜI GIAN (Tích lũy sinh học liên tục qua mỗi 5 giây)
        fall_duration = 0
        for i in range(rows_per_event):
            current_time = base_time + timedelta(seconds=i * 5)
            accel = random.uniform(0.95, 1.05) + (max(0.0, exertion) * 0.1)

            # Cập nhật các biến cố động theo thời gian của kịch bản đặc thù
            if scenario_name == 'NORMAL_JUMPING' and i in [20, 50, 80]:
                accel = random.uniform(3.6, 5.2)  # Xảy ra các cú nhảy bất ngờ (Nhiễu gia tốc)

            elif scenario_name == 'WARN_INTERNAL_INJURY':
                if i == 0: accel = random.uniform(4.5, 6.0)  # Cú ngã ban đầu
                trauma_shock = min(0.6, trauma_shock + 0.008)  # Xuất huyết nội tăng dần cú sốc qua thời gian

            elif scenario_name == 'CRIT_SEVERE_FALL':
                if i == 0: accel = random.uniform(6.5, 9.0)  # Ngã chấn thương chấn động mạnh
                fall_duration = i * 5
                trauma_shock = min(1.0, trauma_shock + 0.02)  # Sốc chấn thương chuyển biến nặng rất nhanh

            # --- KHỐI GIẢI PHƯƠNG TRÌNH SINH LÝ HỌC (Physiological Core) ---
            # A. Động lực học Thân nhiệt (Nhiệt nội sinh do vận động VS Trao đổi nhiệt môi trường)
            heat_production = 0.012 * (1.0 + exertion)
            heat_loss_gain = 0.004 * (env_temp - body_temp) * (1.0 - clothing)
            body_temp += heat_production + heat_loss_gain + random.uniform(-0.01, 0.01)
            body_temp = np.clip(body_temp, 30.0, 43.0)

            # B. Động lực học Nhịp tim (Phụ thuộc vào Vận động, Thiếu Oxy, Nhiễm độc và Sốc chấn thương)
            hypoxia_demand = max(0.0, 20.9 - oxygen_env) * 4.5
            toxic_demand = max(0.0, aqi_env - 100.0) * 0.04
            # Sốc chấn thương giai đoạn đầu làm tim đập nhanh bù máu, nhưng nếu ngất lịm (exertion < 0) tim sẽ suy sụp hạ dần
            shock_demand = trauma_shock * 40.0 if exertion >= 0 else -trauma_shock * 25.0

            target_hr = base_hr + (exertion * 65.0) + hypoxia_demand + toxic_demand + shock_demand
            hr += (target_hr - hr) * 0.12 + random.uniform(-1.2, 1.2)  # Quán tính tim mạch
            hr = np.clip(hr, 35.0, 195.0)

            # C. Động lực học Chỉ số HRV (Phản ánh độ căng thẳng/suy kiệt của hệ thần kinh tự chủ)
            thermal_stress = abs(body_temp - base_temp) * 16.0
            hypoxia_stress = max(0.0, 20.9 - oxygen_env) * 10.0
            toxic_stress = max(0.0, aqi_env - 50.0) * 0.12
            injury_stress = trauma_shock * 45.0

            target_hrv = base_hrv - (exertion * 20.0) - thermal_stress - hypoxia_stress - toxic_stress - injury_stress
            hrv += (target_hrv - hrv) * 0.08 + random.uniform(-0.8, 0.8)
            hrv = np.clip(hrv, 4.0, 100.0)

            # --- BỘ PHÂN LOẠI LÂM SÀNG TỰ ĐỘNG (Clinical Risk Triage Labeler) ---
            # Nhãn sinh ra hoàn toàn dựa trên mức độ tàn phá/sai lệch của các vector sinh lý thực tế tại giây đó
            biological_deviance = 0.0
            biological_deviance += (abs(body_temp - 36.8) / 1.3) ** 2  # Sai lệch thân nhiệt
            biological_deviance += (max(0.0, 20.9 - oxygen_env) / 2.0) ** 2  # Suy giảm Oxy hoại tử
            biological_deviance += (max(0.0, aqi_env - 120.0) / 100.0) ** 2  # Độc tính máu
            biological_deviance += (max(0.0, 52.0 - hrv) / 14.0) ** 2  # Suy kiệt thần kinh tự chủ
            if fall_duration > 15: biological_deviance += (fall_duration / 25.0)  # Thời gian bất tỉnh
            if trauma_shock > 0.2: biological_deviance += (trauma_shock * 3.5)

            # Phân định nhãn sinh học dựa trên tổng điểm nguy cơ tích lũy liên tục
            if biological_deviance >= 4.8:
                derived_label = 2  # Nguy kịch (Critical)
            elif biological_deviance >= 1.4:
                derived_label = 1  # Cảnh báo rủi ro ngầm (Caution)
            else:
                derived_label = 0  # Hoàn toàn an toàn hoặc thích nghi sinh học tốt (Safe)

            data.append([
                current_time, user_id,
                round(hr, 1), round(hrv, 1), round(body_temp, 2),
                round(env_temp, 1), round(humidity, 1), round(pressure, 1), round(oxygen_env, 2),
                round(uv_index, 1), int(aqi_env), round(accel, 2), fall_duration, derived_label
            ])

    return data


if __name__ == "__main__":
    print("=== BẮT ĐẦU KHỞI TẠO DATASET PHẢN HỒI SINH LÝ LIÊN TỤC V5 ===")
    all_data = []

    # 1. NHÓM AN TOÀN (Gồm cả nhiễu tim cao khi leo núi nặng và nhảy cao không ngã)
    print("-> Đang mô phỏng các trạng thái vận động bình thường và thích nghi nhiệt...")
    all_data.extend(simulate_user_timeline('NORMAL_WALKING', 65, 100, 0))
    all_data.extend(simulate_user_timeline('NORMAL_HEAVY_CLIMB', 20, 100, 0))
    all_data.extend(simulate_user_timeline('NORMAL_JUMPING', 15, 100, 0))

    # 2. NHÓM CẢNH BÁO (Rủi ro ngầm tiến triển chậm)
    print("-> Đang mô phỏng các tiến trình rủi ro sinh học ngầm...")
    all_data.extend(simulate_user_timeline('WARN_INTERNAL_INJURY', 20, 50, 1))
    all_data.extend(simulate_user_timeline('WARN_SILENT_POISON', 20, 50, 1))
    all_data.extend(simulate_user_timeline('WARN_SLEEP_HYPOTHERMIA', 20, 50, 1))

    # 3. NHÓM NGUY KỊCH (Tai nạn cấp tính hoặc biến chứng sốc nặng)
    print("-> Đang mô phỏng các pha suy sụp sinh học cấp tính...")
    all_data.extend(simulate_user_timeline('CRIT_SEVERE_FALL', 10, 50, 2))
    all_data.extend(simulate_user_timeline('CRIT_HEATSTROKE', 20, 50, 2))

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'user_id', 'hr', 'hrv', 'body_temp', 'env_temp',
        'humidity', 'pressure', 'oxygen', 'uv_index', 'aqi',
        'accel_mag', 'fall_duration', 'label'
    ])

    output_filename = "simulation_survival_dataset5.csv"
    df.to_csv(output_filename, index=False)

    print(f"\n✅ Đã xuất tập dữ liệu chất lượng cao: {output_filename}")
    print("📊 Phân bổ nhãn thực tế thu được thông qua Bộ chấm điểm sinh học:")
    print(df['label'].value_counts().sort_index())