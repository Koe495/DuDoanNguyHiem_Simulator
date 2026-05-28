import streamlit as st
import pandas as pd
import numpy as np
import random
import time

# ==========================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Thám Hiểm Sinh Tồn AI", page_icon="🏔️", layout="wide")

# ==========================================
# KHỞI TẠO STATE CỦA GAME (Lưu trạng thái)
# ==========================================
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'MENU'  # MENU, PLAYING, GAME_OVER
if 'tick' not in st.session_state:
    st.session_state.tick = 0
if 'danger_ticks' not in st.session_state:
    st.session_state.danger_ticks = 0  # Đếm thời gian chìm trong nguy hiểm
if 'player' not in st.session_state:
    st.session_state.player = {}
if 'env' not in st.session_state:
    st.session_state.env = {}


# ==========================================
# HÀM MÔ PHỎNG MÔ HÌNH HỌC MÁY (ML ENGINE)
# ==========================================
def ml_predict_risk(player, env):
    """
    Trong thực tế, bạn sẽ dùng:
    model = joblib.load('random_forest_model.pkl')
    return model.predict(df)[0]
    Ở đây tôi viết một hàm logic mô phỏng lại cách cây quyết định (Random Forest) hoạt động.
    """
    risk_score = 0
    # Tính đặc trưng (Feature Engineering)
    hr_diff = player['hr'] - player['base_hr']

    # Cây quyết định 1: Hạ thân nhiệt
    if env['temp'] < 10 and player['body_temp'] < 36.0:
        risk_score += 2
    # Cây quyết định 2: Sốc nhiệt
    if env['temp'] > 38 and hr_diff > 40:
        risk_score += 2
    # Cây quyết định 3: Thiếu oxy
    if env['oxygen'] < 16.0:
        risk_score += 2
    # Cây quyết định 4: Chấn thương té ngã
    if player['fall_duration'] > 15:
        risk_score += 3

    if risk_score >= 2:
        return 2  # Nguy kịch
    elif risk_score == 1:
        return 1  # Cảnh báo
    return 0  # An toàn


# ==========================================
# MÀN HÌNH 1: MENU CHÍNH
# ==========================================
def draw_menu():
    st.title("🏔️ TRÒ CHƠI MÔ PHỎNG: THÁM HIỂM SINH TỒN AI")
    st.markdown("Nhập thông số sinh lý cơ bản (Baseline) của bạn để hệ thống AI hiệu chỉnh.")

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Tên người chơi:", "Nhà Thám Hiểm")
        base_hr = st.slider("Nhịp tim cơ bản (bpm):", 50, 100, 75)
    with col2:
        base_temp = st.slider("Thân nhiệt cơ bản (°C):", 36.0, 37.5, 36.8)

    if st.button("🚀 BẮT ĐẦU THÁM HIỂM", use_container_width=True):
        # Khởi tạo thông số người chơi
        st.session_state.player = {
            'name': player_name,
            'base_hr': base_hr,
            'base_temp': base_temp,
            'hr': base_hr,
            'body_temp': base_temp,
            'fall_duration': 0,
            'has_coat': False,
            'has_oxygen_mask': False,
            'status_emoji': '🚶‍♂️'
        }
        # Khởi tạo thông số môi trường (Lý tưởng)
        st.session_state.env = {
            'temp': 24.0,
            'oxygen': 21.0
        }
        st.session_state.game_state = 'PLAYING'
        st.session_state.tick = 0
        st.session_state.danger_ticks = 0
        st.rerun()


# ==========================================
# VÒNG LẶP LOGIC GAME
# ==========================================
def game_tick():
    p = st.session_state.player
    e = st.session_state.env

    st.session_state.tick += 1

    # 1. Sinh dữ liệu (Random Walk & Logic)
    if p['fall_duration'] > 0:
        p['fall_duration'] += 5  # Mỗi tick = 5 giây
        p['hr'] = max(40, p['hr'] - random.uniform(1, 3))  # Tim lịm đi
        p['status_emoji'] = '🤕 (Té ngã)'
    else:
        # Nhịp tim biến thiên ngẫu nhiên
        p['hr'] = p['base_hr'] + random.uniform(-3, 5)
        p['status_emoji'] = '🚶‍♂️ (Đang đi)'

        # Môi trường tác động
        if e['temp'] < 10 and not p['has_coat']:
            p['body_temp'] -= random.uniform(0.1, 0.3)
            p['status_emoji'] = '🥶 (Lạnh run)'
        elif e['temp'] > 35:
            p['body_temp'] += random.uniform(0.1, 0.3)
            p['hr'] += random.uniform(10, 20)
            p['status_emoji'] = '🥵 (Sốc nhiệt)'

        if e['oxygen'] < 18 and not p['has_oxygen_mask']:
            p['hr'] += random.uniform(15, 30)
            p['status_emoji'] = '😵 (Thiếu oxy)'

    # 2. Gọi AI dự đoán nguy hiểm
    risk_level = ml_predict_risk(p, e)

    # 3. Xử lý kết quả AI (Luật chết)
    if risk_level == 2:
        st.session_state.danger_ticks += 1
        if st.session_state.danger_ticks >= 5:  # Chịu đựng quá 5 Ticks (25 giây) thì Game Over
            st.session_state.game_state = 'GAME_OVER'
    else:
        st.session_state.danger_ticks = max(0, st.session_state.danger_ticks - 1)

    st.session_state.current_risk = risk_level


# ==========================================
# MÀN HÌNH 2: TRÒ CHƠI
# ==========================================
def draw_game():
    st.title(f"📍 Hành trình của {st.session_state.player['name']} - Giây thứ: {st.session_state.tick * 5}")

    # KHU VỰC CẢNH BÁO AI
    risk = st.session_state.get('current_risk', 0)
    if risk == 0:
        st.success("🤖 AI Nhận định: TÌNH TRẠNG AN TOÀN")
    elif risk == 1:
        st.warning("🤖 AI Nhận định: CÓ DẤU HIỆU NGUY CƠ (Cảnh báo cam)")
    else:
        st.error(
            f"🤖 AI Nhận định: NGUY KỊCH! Sắp tử vong nếu không xử lý (Đã chịu đựng: {st.session_state.danger_ticks}/5)")

    col_ui, col_controls = st.columns([2, 1])

    with col_ui:
        st.markdown("### Giao diện Camera hành trình")
        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{st.session_state.player['status_emoji']}</h1>",
                    unsafe_allow_html=True)

        # Bảng thông số sinh lý do thiết bị đeo tay thu thập
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Nhịp tim (HR)", f"{int(st.session_state.player['hr'])} bpm")
        col_s2.metric("Thân nhiệt", f"{st.session_state.player['body_temp']:.1f} °C")
        col_s3.metric("Té ngã", f"{st.session_state.player['fall_duration']} giây")

    with col_controls:
        st.markdown("### 🌍 Can thiệp Môi trường")
        # Thay đổi môi trường
        env_temp = st.slider("Nhiệt độ (°C)", -20.0, 50.0, st.session_state.env['temp'])
        env_oxy = st.slider("Nồng độ Oxy (%)", 10.0, 21.0, st.session_state.env['oxygen'])
        st.session_state.env['temp'] = env_temp
        st.session_state.env['oxygen'] = env_oxy

        st.markdown("### 🎒 Hành động")
        # Các nút tương tác
        if st.button("💥 Bị đẩy ngã (Tai nạn)"):
            st.session_state.player['fall_duration'] = 1  # Kích hoạt ngã

        if st.button("🧥 Mặc áo ấm (Chống lạnh)"):
            st.session_state.player['has_coat'] = True

        if st.button("🤿 Đeo bình Oxy"):
            st.session_state.player['has_oxygen_mask'] = True

        if st.button("🚑 Đứng dậy / Cấp cứu (Hồi phục)"):
            st.session_state.player['fall_duration'] = 0
            st.session_state.player['body_temp'] = st.session_state.player['base_temp']
            st.session_state.danger_ticks = 0

    st.markdown("---")
    # Nút Tick (Tiến tới 5 giây) để Game hoạt động
    if st.button("⏭️ TIẾN TỚI 5 GIÂY (Next Tick)", use_container_width=True, type="primary"):
        game_tick()
        st.rerun()


# ==========================================
# MÀN HÌNH 3: GAME OVER
# ==========================================
def draw_game_over():
    st.markdown("<h1 style='text-align: center; font-size: 100px;'>💀</h1>", unsafe_allow_html=True)
    st.error("GAME OVER! Người thám hiểm đã không thể sinh tồn do các chỉ số y sinh vượt quá ngưỡng chịu đựng.")
    if st.button("🔄 Chơi lại từ đầu", use_container_width=True):
        st.session_state.game_state = 'MENU'
        st.rerun()


# ==========================================
# ĐIỀU HƯỚNG MÀN HÌNH
# ==========================================
if st.session_state.game_state == 'MENU':
    draw_menu()
elif st.session_state.game_state == 'PLAYING':
    draw_game()
elif st.session_state.game_state == 'GAME_OVER':
    draw_game_over()