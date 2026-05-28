import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# BƯỚC 1 & BƯỚC 2: SIMULATOR & ANOMALY INJECTOR
# ==========================================

@st.cache_resource
def generate_and_train_model():
    """
    Hàm giả lập sinh 10,000 dòng dữ liệu bình thường (Bước 1)
    và inject 2,000 dòng dữ liệu bất thường/nguy hiểm (Bước 2)
    sau đó huấn luyện mô hình Random Forest để sử dụng trong game.
    """
    np.random.seed(42)

    # --- Bước 1: Sinh 10,000 dòng dữ liệu "Bình thường" (Nhãn 0) ---
    n_normal = 10000
    normal_data = {
        'HR': np.random.normal(75, 5, n_normal),  # Nhịp tim bình thường 65-85
        'HRV': np.random.normal(70, 10, n_normal),  # Biến thiên nhịp tim ổn định
        'Body_Temp': np.random.normal(36.8, 0.2, n_normal),  # Thân nhiệt bình thường
        'Fall_Detected': np.zeros(n_normal),  # Không ngã
        'Fall_Duration': np.zeros(n_normal),
        'Ambient_Temp': np.random.uniform(20, 28, n_normal),  # Thời tiết lý tưởng
        'Humidity': np.random.uniform(40, 60, n_normal),
        'Pressure': np.random.uniform(1005, 1015, n_normal),
        'Oxygen_Level': np.random.uniform(20.5, 21.0, n_normal),  # Oxy dồi dào
        'UV_Index': np.random.uniform(1, 3, n_normal),
        'AQI': np.random.uniform(15, 50, n_normal),  # Không khí trong lành
        'Label': np.zeros(n_normal)  # 0: An toàn
    }
    df_normal = pd.DataFrame(normal_data)

    # --- Bước 2: Chèn 2,000 dòng dữ liệu "Nguy hiểm" (Nhãn 1: Nguy cơ, 2: Nguy hiểm) ---
    n_anomaly = 2000
    anomaly_rows = []

    scenarios = ['sốc nhiệt', 'hạ thân nhiệt', 'hiếm khí', 'té ngã', 'ngộ độc khí']

    for _ in range(n_anomaly):
        scenario = np.random.choice(scenarios)
        # Khởi tạo mặc định giống bình thường trước khi biến đổi đột biến
        row = {
            'HR': np.random.normal(75, 5), 'HRV': np.random.normal(70, 10),
            'Body_Temp': np.random.normal(36.8, 0.2), 'Fall_Detected': 0.0, 'Fall_Duration': 0.0,
            'Ambient_Temp': np.random.uniform(20, 28), 'Humidity': np.random.uniform(40, 60),
            'Pressure': np.random.uniform(1005, 1015), 'Oxygen_Level': np.random.uniform(20.5, 21.0),
            'UV_Index': np.random.uniform(1, 3), 'AQI': np.random.uniform(15, 50),
            'Label': 2.0  # Mặc định các ca nặng là Nguy cơ cao (2)
        }

        if scenario == 'sốc nhiệt':
            row['Ambient_Temp'] = np.random.uniform(40, 50)
            row['HR'] = np.random.uniform(120, 160)
            row['Body_Temp'] = np.random.uniform(39.0, 41.0)
            row['HRV'] = np.random.uniform(20, 40)
        elif scenario == 'hạ thân nhiệt':
            row['Ambient_Temp'] = np.random.uniform(-20, -5)
            row['Body_Temp'] = np.random.uniform(33.0, 35.0)
            row['HR'] = np.random.uniform(45, 60)
        elif scenario == 'hiếm khí':
            row['Oxygen_Level'] = np.random.uniform(10, 14)
            row['Pressure'] = np.random.uniform(600, 750)
            row['HR'] = np.random.uniform(110, 140)
            row['HRV'] = np.random.uniform(25, 45)
        elif scenario == 'té ngã':
            row['Fall_Detected'] = 1.0
            row['Fall_Duration'] = np.random.uniform(10, 60)
            row['HR'] = np.random.uniform(100, 130)
        elif scenario == 'ngộ độc khí':
            row['AQI'] = np.random.uniform(250, 500)
            row['HR'] = np.random.uniform(100, 130)
            row['Oxygen_Level'] = np.random.uniform(16, 19)

        anomaly_rows.append(row)

    df_anomaly = pd.DataFrame(anomaly_rows)

    # Trộn tập dữ liệu xáo trộn rải rác
    df_final = pd.concat([df_normal, df_anomaly], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Huấn luyện mô hình Học máy để dùng cho Game
    X = df_final.drop(columns=['Label'])
    y = df_final['Label']

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    return model


# Khởi tạo mô hình học máy từ dữ liệu mô phỏng
ml_model = generate_and_train_model()

# ==========================================
# BƯỚC 3: TẠO GAME MÔ PHỎNG VỚI STREAMLIT
# ==========================================

st.set_page_config(page_title="Hệ thống Sinh tồn Thám hiểm ML", layout="wide")

# Quản lý trạng thái màn hình (Menu / Game)
if 'page' not in st.session_state:
    st.session_state.page = 'menu'

# --- MÀN HÌNH MENU ---
if st.session_state.page == 'menu':
    st.title("🏕️ Game Mô Phỏng Sinh Tồn Thám Hiểm (AI Risk Predictor)")
    st.write("Chào mừng bạn đến với hệ thống dự đoán mức độ nguy hiểm dựa trên dữ liệu thiết bị đeo tay.")

    with st.form("menu_form"):
        player_name = st.text_input("Nhập tên người chơi:", value="Người thám hiểm X")

        st.subheader("Thông số sức khỏe cơ bản ban đầu (Baseline)")
        base_hr = st.slider("Nhịp tim trung bình khi khỏe mạnh (BPM):", min_value=60, max_value=90, value=75)
        base_temp = st.slider("Thân nhiệt cơ bản (°C):", min_value=36.0, max_value=37.5, value=36.8, step=0.1)

        submit = st.form_submit_button("Bắt đầu hành trình")

        if submit:
            if not player_name.strip():
                st.error("Vui lòng điền tên người chơi!")
            else:
                st.session_state.player_name = player_name
                st.session_state.base_hr = base_hr
                st.session_state.base_temp = base_temp

                # Khởi tạo thông số môi trường ban đầu (Lý tưởng)
                st.session_state.env_target = {"temp": 25.0, "oxygen": 20.9, "aqi": 25.0, "pressure": 1013.0, "uv": 2.0,
                                               "humidity": 50.0}
                st.session_state.env_current = {"temp": 25.0, "oxygen": 20.9, "aqi": 25.0, "pressure": 1013.0,
                                                "uv": 2.0, "humidity": 50.0}

                # Khởi tạo trạng thái nhân vật và tương tác
                st.session_state.is_fallen = False
                st.session_state.fall_time = 0.0
                st.session_state.has_jacket = False
                st.session_state.has_mask = False
                st.session_state.health = 100
                st.session_state.danger_duration = 0  # Thời gian liên tục ở mức nguy hiểm

                st.session_state.page = 'game'
                st.rerun()

# --- MÀN HÌNH TRÒ CHƠI ---
elif st.session_state.page == 'game':
    st.title(f"🏃 Cuộc phiêu lưu sinh tồn của: {st.session_state.player_name}")

    # Thiết lập các Địa hình / Thời tiết sẵn có (Presets)
    st.subheader("🗺️ Lựa chọn địa hình di chuyển đến:")
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("🏔️ Núi Tuyết Độ Cao (Lạnh, Thiếu Oxy)"):
        st.session_state.env_target = {"temp": -15.0, "oxygen": 13.5, "aqi": 10.0, "pressure": 700.0, "uv": 8.0,
                                       "humidity": 25.0}
    if col2.button("🌋 Hang Sâu Khí Độc (Hiếm khí, AQI tệ)"):
        st.session_state.env_target = {"temp": 35.0, "oxygen": 12.0, "aqi": 320.0, "pressure": 1040.0, "uv": 0.0,
                                       "humidity": 85.0}
    if col3.button("🏜️ Sa Mạc Khắc Nghiệt (Nóng, UV cao)"):
        st.session_state.env_target = {"temp": 46.0, "oxygen": 20.5, "aqi": 75.0, "pressure": 1002.0, "uv": 11.0,
                                       "humidity": 10.0}
    if col4.button("🏡 Thung Lũng Lý Tưởng (An toàn)"):
        st.session_state.env_target = {"temp": 25.0, "oxygen": 20.9, "aqi": 25.0, "pressure": 1013.0, "uv": 2.0,
                                       "humidity": 50.0}

    st.markdown("---")

    # Cập nhật thông số môi trường DẦN DẦN (Mô phỏng đang di chuyển tới vùng đất mới)
    for key in st.session_state.env_current.keys():
        diff = st.session_state.env_target[key] - st.session_state.env_current[key]
        st.session_state.env_current[key] += diff * 0.25  # Tiến dần 25% mỗi chu kỳ làm mới

    # Tính toán logic biến đổi SỨC KHỎE dựa trên môi trường và hành động tương tác
    cur_env = st.session_state.env_current

    # Tính toán các biến đổi sinh lý thực tế dựa trên tác động xung quanh
    calc_hr = st.session_state.base_hr
    calc_temp = st.session_state.base_temp
    calc_hrv = 70.0

    # Tác động của nhiệt độ
    if cur_env['temp'] > 38:
        if not st.session_state.has_jacket:  # Không cởi bớt áo (hoặc đang mặc bình thường ở môi trường quá nóng)
            calc_temp += (cur_env['temp'] - 38) * 0.1
            calc_hr += (cur_env['temp'] - 38) * 4
            calc_hrv -= 15
    elif cur_env['temp'] < 5:
        if not st.session_state.has_jacket:  # Thiếu áo ấm khi trời lạnh
            calc_temp -= (5 - cur_env['temp']) * 0.15
            calc_hr -= (5 - cur_env['temp']) * 1.5
            calc_hrv -= 10

    # Tác động của việc thiếu Oxy hoặc khói độc
    effective_oxygen = 20.9 if st.session_state.has_mask else cur_env['oxygen']
    if effective_oxygen < 16:
        calc_hr += (16 - effective_oxygen) * 8
        calc_hrv -= (16 - effective_oxygen) * 5

    if cur_env['aqi'] > 150 and not st.session_state.has_mask:
        calc_hr += (cur_env['aqi'] - 150) * 0.1

    # Tác động té ngã
    if st.session_state.is_fallen:
        st.session_state.fall_time += 2.0
        calc_hr += 20
        calc_hrv -= 20
    else:
        st.session_state.fall_time = 0.0

    # Gom toàn bộ thông số đầu vào cho Mô hình AI Học Máy dự đoán
    input_features = pd.DataFrame([{
        'HR': max(40.0, min(180.0, calc_hr)),
        'HRV': max(5.0, min(120.0, calc_hrv)),
        'Body_Temp': max(30.0, min(43.0, calc_temp)),
        'Fall_Detected': 1.0 if st.session_state.is_fallen else 0.0,
        'Fall_Duration': st.session_state.fall_time,
        'Ambient_Temp': cur_env['temp'],
        'Humidity': cur_env['humidity'],
        'Pressure': cur_env['pressure'],
        'Oxygen_Level': cur_env['oxygen'],
        'UV_Index': cur_env['uv'],
        'AQI': cur_env['aqi']
    }])

    # Dự đoán bằng Mô hình ML đã huấn luyện từ dữ liệu Simulator
    pred_risk = ml_model.predict(input_features)[0]

    # Phân chia bố cục hiển thị Game
    col_char, col_stats, col_control = st.columns([1, 2, 2])

    with col_char:
        st.subheader("🧍 Nhân vật")
        # Thay đổi hình nhân tùy thuộc vào trạng thái sức khỏe
        if st.session_state.health <= 0:
            st.markdown("<h1 style='text-align: center; font-size: 80px;'>💀</h1>", unsafe_allow_html=True)
            st.error("TRẠNG THÁI: TỬ VONG")
        elif st.session_state.is_fallen:
            st.markdown("<h1 style='text-align: center; font-size: 80px;'>🧎</h1>", unsafe_allow_html=True)
            st.warning("TRẠNG THÁI: BỊ NGÃ!")
        elif pred_risk == 2:
            st.markdown("<h1 style='text-align: center; font-size: 80px;'>🏃🆘</h1>", unsafe_allow_html=True)
            st.error("TRẠNG THÁI: NGUY CƠ CAO")
        elif pred_risk == 1:
            st.markdown("<h1 style='text-align: center; font-size: 80px;'>🏃⚠️</h1>", unsafe_allow_html=True)
            st.warning("TRẠNG THÁI: NGUY CƠ TRUNG BÌNH")
        else:
            st.markdown("<h1 style='text-align: center; font-size: 80px;'>🚶✨</h1>", unsafe_allow_html=True)
            st.success("TRẠNG THÁI: AN TOÀN")

        st.metric(label="Máu (Health)", value=f"{st.session_state.health}/100")

    with col_stats:
        st.subheader("📊 Dữ liệu thiết bị đeo tay thu thập")
        st.write(f"**Nhịp tim (HR):** {input_features['HR'].values[0]:.1f} BPM")
        st.write(f"**Biến thiên nhịp tim (HRV):** {input_features['HRV'].values[0]:.1f} ms")
        st.write(f"**Thân nhiệt cơ thể:** {input_features['Body_Temp'].values[0]:.2f} °C")
        st.write(f"**Nhiệt độ môi trường:** {input_features['Ambient_Temp'].values[0]:.1f} °C")
        st.write(f"**Nồng độ Oxy khí quyển:** {input_features['Oxygen_Level'].values[0]:.1f} %")
        st.write(f"**Chỉ số AQI không khí:** {input_features['AQI'].values[0]:.1f}")

    with col_control:
        st.subheader("🎮 Hành động / Tương tác chuột")

        # Nút tương tác hất ngã nhân vật
        if not st.session_state.is_fallen:
            if st.button("💥 Hất ngã nhân vật"):
                st.session_state.is_fallen = True
                st.rerun()
        else:
            if st.button("💪 Đứng dậy"):
                st.session_state.is_fallen = False
                st.rerun()

        # Nút quản lý trang phục
        if cur_env['temp'] < 10:
            if st.button("🧥 Mặc thêm quần áo ấm" if not st.session_state.has_jacket else "🧥 Cởi bớt áo khoác dày"):
                st.session_state.has_jacket = not st.session_state.has_jacket
                st.rerun()
        else:
            if st.button(
                    "👕 Cởi bớt quần áo cho mát" if not st.session_state.has_jacket else "👕 Mặc lại trang phục thường"):
                st.session_state.has_jacket = not st.session_state.has_jacket
                st.rerun()

        # Nút đeo mặt nạ khí độc / dưỡng khí oxy
        if st.button("🤿 Đeo mặt nạ dưỡng khí" if not st.session_state.has_mask else "🤿 Tháo mặt nạ dưỡng khí"):
            st.session_state.has_mask = not st.session_state.has_mask
            st.rerun()

    # --- HỆ THỐNG CẢNH BÁO NGUY HIỂM ---
    st.subheader("🚨 Bảng điều khiển Cảnh báo AI (Thời gian thực)")

    # Hiển thị các cảnh báo chi tiết dựa trên tình huống cụ thể mà mô hình ML đang quét thấy bất thường
    has_alert = False
    if input_features['Body_Temp'].values[0] > 38.5:
        st.error("⚠️ CẢNH BÁO: Phát hiện sốc nhiệt cao! Cần cởi bỏ trang phục dày lập tức.")
        has_alert = True
    if input_features['Body_Temp'].values[0] < 35.0:
        st.error("⚠️ CẢNH BÁO: Cơ thể đang hạ thân nhiệt cấp độ nguy hiểm! Cần mặc áo ấm ngay.")
        has_alert = True
    if cur_env['oxygen'] < 15.0 and not st.session_state.has_mask:
        st.error("⚠️ CẢNH BÁO: Môi trường thiếu hụt Oxy trầm trọng! Hãy kích hoạt mặt nạ dưỡng khí.")
        has_alert = True
    if cur_env['aqi'] > 200 and not st.session_state.has_mask:
        st.error("⚠️ CẢNH BÁO: Nồng độ khói/khí độc hại vượt ngưỡng! Đeo mặt nạ phòng độc khẩn cấp.")
        has_alert = True
    if st.session_state.is_fallen and st.session_state.fall_time > 5.0:
        st.error(
            f"⚠️ CẢNH BÁO: Phát hiện chấn thương do té ngã kéo dài ({st.session_state.fall_time:.0f}s)! Mau đứng dậy.")
        has_alert = True

    if not has_alert and pred_risk == 0:
        st.info("✅ Mọi thông số sinh tồn đang nằm trong vùng an toàn lý tưởng.")

    # --- XỬ LÝ TRẠNG THÁI TỬ VONG / GAME OVER ---
    if pred_risk == 2 or has_alert:
        st.session_state.danger_duration += 1
        # Trừ máu dần theo thời gian nếu không kịp thời ứng phó xử lý
        if st.session_state.danger_duration > 2:
            st.session_state.health = max(0, st.session_state.health - 20)
    else:
        st.session_state.danger_duration = 0
        st.session_state.health = min(100, st.session_state.health + 5)  # Hồi phục nhẹ nếu an toàn

    # Nếu chết, dừng game và hiển thị nút Reset khởi động lại trò chơi
    if st.session_state.health <= 0:
        st.error("💀 BẠN ĐÃ TỬ VONG DO KHÔNG KỊP THỜI XỬ LÝ CÁC NGUY CƠ SINH TỒN!")
        if st.button("🔄 Khởi động lại trò chơi"):
            st.session_state.page = 'menu'
            st.rerun()
    else:
        # Tự động làm mới giao diện sau mỗi 2 giây để cập nhật dữ liệu liên tục chuỗi thời gian
        time.sleep(2.0)
        st.rerun()