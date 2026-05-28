import random
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. KHỞI TẠO & HUẤN LUYỆN MÔ HÌNH HỌC MÁY THẬT
# ==========================================
@st.cache_resource
def load_and_train_ml_model():
    # Tìm file dataset của bạn
    file_name = 'multi_user_survival_dataset2.csv'
    if not os.path.exists(file_name):
        file_name = 'multi_user_survival_dataset.csv'  # Fallback dự phòng

    if not os.path.exists(file_name):
        st.error(f"❌ LỖI: Không tìm thấy file '{file_name}'. Vui lòng sinh dữ liệu trước!")
        st.stop()

    df = pd.read_csv(file_name)

    # Thực hiện Feature Engineering giống lúc huấn luyện
    df['hr_baseline'] = df.groupby('user_id')['hr'].transform('mean')
    df['hr_diff_from_baseline'] = df['hr'] - df['hr_baseline']
    df['hr_rolling_std'] = df.groupby('user_id')['hr'].transform(lambda x: x.rolling(window=12, min_periods=1).std())
    df['hr_rolling_std'].fillna(0, inplace=True)

    # Các đặc trưng bắt buộc phải có để đưa vào mô hình
    features = ['hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv',
                'body_temp', 'env_temp', 'humidity', 'pressure',
                'oxygen', 'uv_index', 'aqi', 'accel_mag', 'fall_duration']

    X = df[features]
    y = df['is_anomaly']

    # Huấn luyện mô hình Random Forest Thật
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model, features


# Tải mô hình vào bộ nhớ
rf_model, expected_features = load_and_train_ml_model()

# ==========================================
# 2. KHỞI TẠO TRẠNG THÁI GAME (STATE)
# ==========================================
st.set_page_config(page_title="Survival ML Game", layout="wide")

if 'game_state' not in st.session_state:
    st.session_state.game_state = 'MENU'
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = True
if 'speed_x2' not in st.session_state:
    st.session_state.speed_x2 = False
if 'history' not in st.session_state:
    st.session_state.history = []  # Lưu lịch sử để phục vụ tính Toán học máy và Tua ngược (Rewind)
if 'target_env' not in st.session_state:
    st.session_state.target_env = {'temp': 24.0, 'oxygen': 20.9, 'pressure': 1013.0, 'humidity': 55.0, 'aqi': 35}


# ==========================================
# 3. LOGIC THỜI GIAN THỰC & DỰ ĐOÁN HỌC MÁY
# ==========================================
def predict_with_real_ml(current_state):
    """Sử dụng mô hình RF thật để dự đoán trạng thái hiện tại"""
    # Trích xuất đặc trưng Time-series từ lịch sử (Sliding Window)
    recent_hr = [s['hr'] for s in st.session_state.history[-12:]]
    rolling_std = np.std(recent_hr) if len(recent_hr) > 1 else 0.0
    diff_from_base = current_state['hr'] - current_state['base_hr']

    # Tạo DataFrame chứa đúng các cột mà mô hình đã học
    input_data = pd.DataFrame([{
        'hr': current_state['hr'],
        'hr_diff_from_baseline': diff_from_base,
        'hr_rolling_std': rolling_std,
        'hrv': current_state['hrv'],
        'body_temp': current_state['body_temp'],
        'env_temp': current_state['env_temp'],
        'humidity': current_state['humidity'],
        'pressure': current_state['pressure'],
        'oxygen': current_state['oxygen'],
        'uv_index': current_state['uv_index'],
        'aqi': current_state['aqi'],
        'accel_mag': current_state['accel_mag'],
        'fall_duration': current_state['fall_duration']
    }])

    # AI DỰ ĐOÁN THẬT SỰ Ở ĐÂY
    prediction = rf_model.predict(input_data)[0]
    return prediction


def update_game_tick():
    """Hàm này chạy tự động mỗi giây để cập nhật trạng thái"""
    last_state = st.session_state.history[-1].copy()
    new_state = last_state.copy()
    new_state['tick'] += 1

    # Nội suy thời tiết (Thay đổi DẦN DẦN hướng về target_env)
    target = st.session_state.target_env
    new_state['env_temp'] += (target['temp'] - new_state['env_temp']) * 0.1
    new_state['oxygen'] += (target['oxygen'] - new_state['oxygen']) * 0.1
    new_state['pressure'] += (target['pressure'] - new_state['pressure']) * 0.1
    new_state['humidity'] += (target['humidity'] - new_state['humidity']) * 0.1
    new_state['aqi'] += (target['aqi'] - new_state['aqi']) * 0.1

    # Logic sinh lý (Bị tác động bởi môi trường)
    if new_state['fall_duration'] > 0:
        new_state['fall_duration'] += 5
        new_state['accel_mag'] = 1.0  # Nằm im
        new_state['hr'] = max(40, new_state['hr'] - random.uniform(0.5, 1.5))
    else:
        new_state['accel_mag'] = max(0.8, min(1.2, new_state['accel_mag'] + random.uniform(-0.1, 0.1)))  # Đang đi bộ
        # Môi trường lạnh làm giảm thân nhiệt
        if new_state['env_temp'] < 10 and not new_state['has_coat']:
            new_state['body_temp'] -= random.uniform(0.05, 0.15)
        # Thiếu oxy làm nhịp tim tăng bù trừ
        if new_state['oxygen'] < 18 and not new_state['has_oxygen_mask']:
            new_state['hr'] += random.uniform(2, 5)

        # Thêm nhiễu ngẫu nhiên tự nhiên (Random Walk)
        new_state['hr'] += random.uniform(-1, 1)
        new_state['hrv'] += random.uniform(-0.5, 0.5)

    # ĐƯA VÀO MÔ HÌNH HỌC MÁY
    ml_risk = predict_with_real_ml(new_state)
    new_state['ml_risk'] = ml_risk

    # Cập nhật mức độ nguy hiểm kéo dài
    if ml_risk == 1:
        new_state['danger_ticks'] += 1
    else:
        new_state['danger_ticks'] = max(0, new_state['danger_ticks'] - 1)

    # Chết nếu nguy hiểm quá lâu (Ví dụ: 10 Ticks ~ 50 giây)
    if new_state['danger_ticks'] >= 10:
        st.session_state.game_state = 'GAME_OVER'

    st.session_state.history.append(new_state)


# ==========================================
# 4. GIAO DIỆN CHƠI (UI)
# ==========================================
if st.session_state.game_state == 'MENU':
    st.title("⚙️ THIẾT LẬP THÔNG SỐ (BASELINE)")
    base_hr = st.number_input("Nhịp tim trung bình lúc khỏe (bpm):", 50, 100, 75)
    base_temp = st.number_input("Thân nhiệt bình thường (°C):", 35.5, 37.5, 36.8)

    if st.button("BẮT ĐẦU CHƠI (ÁP DỤNG HỌC MÁY)"):
        # Trạng thái gốc
        initial_state = {
            'tick': 0, 'danger_ticks': 0, 'ml_risk': 0,
            'base_hr': base_hr, 'base_temp': base_temp,
            'hr': base_hr, 'hrv': 50.0, 'body_temp': base_temp,
            'env_temp': 24.0, 'humidity': 55.0, 'pressure': 1013.0, 'oxygen': 20.9, 'uv_index': 2.0, 'aqi': 35,
            'accel_mag': 1.0, 'fall_duration': 0,
            'has_coat': False, 'has_oxygen_mask': False
        }
        st.session_state.history = [initial_state]
        st.session_state.game_state = 'PLAYING'
        st.session_state.is_playing = True
        st.rerun()

elif st.session_state.game_state == 'PLAYING':
    current = st.session_state.history[-1]

    # --- BẢNG ĐIỀU KHIỂN THỜI GIAN THỰC ---
    st.write(f"### ⏳ Thời gian chơi: {current['tick'] * 5} giây")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("⏸️ Tạm ngừng" if st.session_state.is_playing else "▶️ Tiếp tục chơi"):
        st.session_state.is_playing = not st.session_state.is_playing
        st.rerun()

    if c2.button("⏪ Tua ngược 5s (Rewind)"):
        if len(st.session_state.history) > 1:
            st.session_state.history.pop()  # Xóa state hiện tại để lùi về trước
            st.session_state.is_playing = False  # Tự động pause khi tua
            st.rerun()

    speed_x2 = c3.checkbox("⏩ x2 Tốc độ", value=st.session_state.speed_x2)
    st.session_state.speed_x2 = speed_x2

    st.markdown("---")

    # --- HIỂN THỊ DỮ LIỆU & HỌC MÁY ---
    col_left, col_right = st.columns([2, 1])
    with col_left:
        # CẢNH BÁO TỪ MÔ HÌNH ML THẬT
        if current['ml_risk'] == 1:
            st.error(
                f"🚨 [MACHINE LEARNING ALERT] HỆ THỐNG PHÁT HIỆN NGUY HIỂM! (Thời gian chịu đựng: {current['danger_ticks']}/10)")
        else:
            st.success("✅ [MACHINE LEARNING ALERT] Mọi chỉ số sinh tồn đang ổn định.")

        # Hình nhân
        emoji = "🤕 Ngã gục" if current['fall_duration'] > 0 else "🚶‍♂️ Đang đi"
        st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)

        c_hr, c_temp, c_fall = st.columns(3)
        c_hr.metric("Nhịp tim (HR)", f"{current['hr']:.1f} bpm", f"{current['hr'] - current['base_hr']:.1f}")
        c_temp.metric("Thân nhiệt", f"{current['body_temp']:.1f} °C")
        c_fall.metric("Thời gian té ngã", f"{current['fall_duration']} s")

    # --- ĐIỀU KHIỂN TƯƠNG TÁC ---
    with col_right:
        st.markdown("#### 🌍 Kịch bản Môi trường")
        if st.button("🏔️ Lên Núi Tuyết"):
            st.session_state.target_env.update({'temp': -15.0, 'oxygen': 16.0, 'pressure': 600.0})
        if st.button("🕳️ Xuống Hang Sâu (Khí độc)"):
            st.session_state.target_env.update({'aqi': 500, 'oxygen': 15.0})
        if st.button("☀️ Về Điều kiện lý tưởng"):
            st.session_state.target_env.update({'temp': 24.0, 'oxygen': 20.9, 'pressure': 1013.0, 'aqi': 35})

        st.markdown("#### 🎒 Tương tác Hình nhân")
        if st.button("💥 Đẩy ngã (Tai nạn)"):
            st.session_state.history[-1]['accel_mag'] = 6.0  # Lực va chạm mạnh
            st.session_state.history[-1]['fall_duration'] = 1
        has_coat = st.checkbox("🧥 Mặc áo ấm", value=current['has_coat'])
        st.session_state.history[-1]['has_coat'] = has_coat
        has_mask = st.checkbox("🤿 Đeo Oxy", value=current['has_oxygen_mask'])
        st.session_state.history[-1]['has_oxygen_mask'] = has_mask

    # ==========================================
    # VÒNG LẶP THỜI GIAN THỰC (REAL-TIME LOOP)
    # ==========================================
    if st.session_state.is_playing:
        # Ngủ để tạo cảm giác thời gian trôi (Nếu x2 thì ngủ ít hơn)
        sleep_time = 0.5 if st.session_state.speed_x2 else 1.0
        time.sleep(sleep_time)

        # Cập nhật logic và gọi hàm st.rerun() để load lại trang
        update_game_tick()
        st.rerun()

elif st.session_state.game_state == 'GAME_OVER':
    st.error("💀 GAME OVER! Mô hình Học máy xác nhận người chơi đã tử vong do vượt quá giới hạn sinh tồn.")
    if st.button("Chơi lại"):
        st.session_state.game_state = 'MENU'
        st.rerun()