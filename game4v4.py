import pygame
import sys
import random
import numpy as np
from collections import deque
import joblib
import pandas as pd

# ==========================================
# 1. CẤU HÌNH CƠ BẢN VÀ MÀU SẮC
# ==========================================
pygame.init()
WIDTH, HEIGHT = 1000, 700
FPS = 60
TICK_RATE = 1000  # 1 giây thực tế = 1 lần cập nhật số liệu

# Bảng màu
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (80, 80, 80)
BLUE = (52, 152, 219)
RED = (231, 76, 60)
ORANGE = (243, 156, 18)
GREEN = (46, 204, 113)
YELLOW = (241, 196, 15)

# Background Môi trường
BG_SNOW = (235, 245, 255)
BG_CAVE = (180, 180, 185)
BG_NORMAL = (245, 255, 245)

font = pygame.font.SysFont('Arial', 20)
large_font = pygame.font.SysFont('Arial', 32, bold=True)

# Load AI Model
try:
    rf_model = joblib.load('rf_model.pkl')
    print("✅ Đã load mô hình AI thành công.")
except:
    rf_model = None
    print("⚠️ Không tìm thấy rf_model.pkl. Chạy chế độ logic dự phòng (Fallback).")


# ==========================================
# 2. LỚP QUẢN LÝ DỮ LIỆU & LOGIC MÔ PHỎNG
# ==========================================
class SimulationData:
    def __init__(self):
        self.age = 25
        self.base_hr = 75.0
        self.base_hrv = 60.0  # Baseline HRV ban đầu

        self.target_env_temp = 24.0
        self.target_oxygen = 20.9
        self.target_pressure = 1013.0
        self.target_aqi = 30.0

        self.env_temp = 24.0
        self.oxygen = 20.9
        self.pressure = 1013.0
        self.aqi = 30.0
        self.humidity = 55.0
        self.uv = 2.0

        self.hr = self.base_hr
        self.hrv = self.base_hrv
        self.body_temp = 36.8
        self.accel = 1.0
        self.fall_duration = 0

        self.hr_history = deque(maxlen=12)

        self.is_sleeping = False
        self.is_heavy_exercise = False
        self.wearing_warm = False
        self.wearing_mask = False
        self.is_fallen = False
        self.auto_stand_timer = 0

        self.ai_status = 0
        self.danger_timer = 0

    def update_environment(self):
        self.env_temp += (self.target_env_temp - self.env_temp) * 0.1
        self.oxygen += (self.target_oxygen - self.oxygen) * 0.1
        self.pressure += (self.target_pressure - self.pressure) * 0.1
        self.aqi += (self.target_aqi - self.aqi) * 0.1

    def update_body(self):
        # 1. LOGIC TÉ NGÃ
        if self.is_fallen:
            if self.auto_stand_timer > 0:
                self.auto_stand_timer -= 1
                if self.auto_stand_timer == 0:
                    self.is_fallen = False
                    self.accel = 1.0
            else:
                self.fall_duration += 5
                self.accel = 1.0
        else:
            self.fall_duration = 0
            if self.is_heavy_exercise:
                self.accel = random.uniform(1.2, 1.8)
            elif self.is_sleeping:
                self.accel = random.uniform(0.9, 1.0)
            else:
                self.accel = random.uniform(0.9, 1.2)

        # 2. LOGIC NHỊP TIM (HR)
        target_hr = self.base_hr
        if self.is_heavy_exercise: target_hr += 60
        if self.is_sleeping: target_hr -= 15
        if self.oxygen < 19 and not self.wearing_mask: target_hr += 15

        self.hr += (target_hr - self.hr) * 0.2 + random.uniform(-1, 1)
        self.hr_history.append(self.hr)

        # 3. LOGIC HRV (BIẾN THIÊN NHỊP TIM - ĐO LƯỜNG MỨC ĐỘ STRESS)
        target_hrv = self.base_hrv

        # Stress thể chất làm giảm HRV
        if self.is_heavy_exercise: target_hrv -= 25
        if self.is_fallen: target_hrv -= 35  # Chấn thương / Đau đớn làm sụt giảm HRV nghiêm trọng

        # Môi trường khắc nghiệt kích thích hệ thần kinh giao cảm -> giảm HRV
        if self.aqi > 150: target_hrv -= 20
        if self.oxygen < 19 and not self.wearing_mask: target_hrv -= 25
        if self.env_temp < 5 and not self.wearing_warm: target_hrv -= 20
        if self.body_temp > 39.0 or self.body_temp < 35.0: target_hrv -= 25  # Sốt/Hạ thân nhiệt gây stress sinh học

        # Ngủ/Nghỉ ngơi kích hoạt hệ phó giao cảm -> tăng HRV (Phục hồi)
        if self.is_sleeping: target_hrv += 20

        # Cập nhật mượt mà
        self.hrv += (target_hrv - self.hrv) * 0.1 + random.uniform(-1, 1)
        self.hrv = np.clip(self.hrv, 5.0, 100.0)

        # 4. LOGIC THÂN NHIỆT
        temp_loss = (24.0 - self.env_temp) * 0.005
        if self.wearing_warm:
            temp_loss -= 0.08
        if self.is_heavy_exercise:
            temp_loss -= 0.05

        self.body_temp -= temp_loss
        self.body_temp = np.clip(self.body_temp + random.uniform(-0.01, 0.01), 30.0, 42.0)

    def get_features_for_ai(self):
        hr_std = np.std(self.hr_history) if len(self.hr_history) > 1 else 0
        return pd.DataFrame([[
            self.hr, self.hr - self.base_hr, hr_std, self.hrv, self.body_temp,
            self.env_temp, self.humidity, self.pressure, self.oxygen,
            self.uv, self.aqi, self.accel, self.fall_duration
        ]], columns=[
            'hr', 'hr_diff_from_baseline', 'hr_rolling_std', 'hrv', 'body_temp',
            'env_temp', 'humidity', 'pressure', 'oxygen', 'uv_index', 'aqi',
            'accel_mag', 'fall_duration'
        ])

    # --- SỬA LỖI TẠI ĐÂY: NÂNG CẤP BỘ LỌC DỰ ĐOÁN LAI (HYBRID) ---
    def predict_danger(self):
        df = self.get_features_for_ai()
        ai_pred = 0
        if rf_model is not None:
            try:
                ai_pred = rf_model.predict(df)[0]
            except:
                ai_pred = 0

        # Luật bảo vệ (Rule-based Overrides) - Đảm bảo luôn Warning và Chết khi chỉ số vượt ngưỡng sinh tồn

        # CẤP ĐỘ 2: NGUY HIỂM CHẾT NGƯỜI (CRITICAL DANGER -> Kích hoạt đếm ngược tử vong)
        if (
                self.body_temp <= 33.5 or self.body_temp >= 41.5 or  # Thân nhiệt quá thấp (đóng băng) hoặc quá cao (sốt co giật)
                self.oxygen <= 14.5 or  # Thiếu oxy trầm trọng (ngạt khí cấp tính)
                self.aqi >= 400.0 or  # Ngộ độc khí độc nồng độ cao
                self.hrv <= 15.0 or  # HRV quá thấp (Hệ thần kinh kiệt quệ, nguy cơ ngừng tim)
                self.hr >= 165.0 or  # Tim đập quá nhanh vượt ngưỡng chịu đựng
                (self.fall_duration >= 20 and self.body_temp < 35.0)  # Ngã chấn thương nặng bất tỉnh giữa trời lạnh
        ):
            return 2

        # CẤP ĐỘ 1: CẢNH BÁO NGUY HIỂM (CAUTION -> Hiển thị hộp thông báo màu Cam)
        if (
                self.body_temp < 35.5 or self.body_temp > 38.8 or  # Chớm hạ thân nhiệt hoặc sốt cao
                self.oxygen < 18.5 or  # Môi trường loãng khí
                self.aqi > 150.0 or  # Không khí ô nhiễm đạt mức độc hại
                self.hrv < 28.0 or  # Chỉ số Stress (HRV) giảm mạnh
                self.hr > 135.0 or  # Tim đập nhanh do quá tải vận động
                (self.accel > 4.0 and self.fall_duration == 0)  # Chấn thương va đập mạnh bất ngờ
        ):
            return max(1, ai_pred)  # Trả về ít nhất là cấp độ 1 nếu thỏa mãn luật bảo vệ

        return ai_pred


# ==========================================
# 3. GIAO DIỆN VÀ MAIN GAME
# ==========================================
class GameApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Explorer Simulator - AI Predicted")
        self.clock = pygame.time.Clock()
        self.state = "MENU"
        self.sim = SimulationData()
        self.last_tick = pygame.time.get_ticks()
        self.is_paused = False

        self.mouse_down_pos = None

        self.active_input = None
        self.input_text = ""
        self.input_rects = {
            'temp': pygame.Rect(WIDTH - 260, 60, 240, 25),
            'oxy': pygame.Rect(WIDTH - 260, 90, 240, 25),
            'aqi': pygame.Rect(WIDTH - 260, 120, 240, 25)
        }

    def draw_button(self, text, x, y, w, h, color, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        rect = pygame.Rect(x, y, w, h)
        is_hover = rect.collidepoint(mouse)

        pygame.draw.rect(self.screen, color if not is_hover else GRAY, rect, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, rect, 2, border_radius=5)

        text_surf = font.render(text, True, BLACK)
        self.screen.blit(text_surf, (x + (w - text_surf.get_width()) // 2, y + (h - text_surf.get_height()) // 2))

        if is_hover and click[0] == 1 and action:
            pygame.time.delay(150)
            action()

    def set_preset_snow(self):
        self.sim.target_env_temp, self.sim.target_oxygen, self.sim.target_aqi = -15.0, 19.0, 10.0

    def set_preset_cave(self):
        self.sim.target_env_temp, self.sim.target_oxygen, self.sim.target_aqi = 15.0, 17.0, 250.0

    def set_preset_normal(self):
        self.sim.target_env_temp, self.sim.target_oxygen, self.sim.target_aqi = 24.0, 20.9, 30.0

    def toggle_clothes(self):
        self.sim.wearing_warm = not self.sim.wearing_warm

    def toggle_mask(self):
        self.sim.wearing_mask = not self.sim.wearing_mask

    def toggle_exercise(self):
        self.sim.is_heavy_exercise = not self.sim.is_heavy_exercise

    def toggle_sleep(self):
        self.sim.is_sleeping = not self.sim.is_sleeping

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.active_input = None

    def start_game(self):
        self.state = "PLAYING"

    def restart_game(self):
        self.sim = SimulationData()
        self.state = "PLAYING"

    def handle_mouse_swipe(self, event):
        player_rect = pygame.Rect(450, 300, 100, 200)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if player_rect.collidepoint(event.pos):
                self.mouse_down_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.mouse_down_pos:
                dx = event.pos[0] - self.mouse_down_pos[0]
                dy = event.pos[1] - self.mouse_down_pos[1]
                speed = np.sqrt(dx ** 2 + dy ** 2)

                if speed > 50:
                    multiplier = 0.4 if self.sim.is_sleeping else 1.0
                    self.sim.accel = (speed / 50.0) * multiplier
                    self.sim.is_fallen = True

                    if self.sim.accel < 3.0:
                        self.sim.auto_stand_timer = 2
                        print(f"Ngã nhẹ ({self.sim.accel:.2f}G). Sẽ tự đứng dậy.")
                    else:
                        self.sim.auto_stand_timer = -1
                        print(f"CHẤN THƯƠNG ({self.sim.accel:.2f}G)! Cần cứu hộ.")

                self.mouse_down_pos = None

    def handle_custom_inputs(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_paused:
            for key, rect in self.input_rects.items():
                if rect.collidepoint(event.pos):
                    self.active_input = key
                    self.input_text = ""
                    return
            self.active_input = None

        if event.type == pygame.KEYDOWN and self.active_input:
            if event.key == pygame.K_RETURN:
                try:
                    val = float(self.input_text)
                    if self.active_input == 'temp':
                        self.sim.target_env_temp = val
                    elif self.active_input == 'oxy':
                        self.sim.target_oxygen = np.clip(val, 0, 100)
                    elif self.active_input == 'aqi':
                        self.sim.target_aqi = max(0, val)
                except ValueError:
                    pass
                self.active_input = None
                self.input_text = ""
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
            else:
                if event.unicode in '0123456789.-':
                    self.input_text += event.unicode

    def draw_player(self):
        color = BLACK if not self.sim.is_fallen else RED
        x, y = 500, 400

        if self.sim.is_fallen:
            pygame.draw.circle(self.screen, color, (x + 50, y + 80), 20)
            pygame.draw.line(self.screen, color, (x + 30, y + 80), (x - 40, y + 80), 5)
            if self.sim.wearing_warm:
                pygame.draw.rect(self.screen, ORANGE, (x - 30, y + 70, 60, 25))
            if self.sim.wearing_mask:
                pygame.draw.circle(self.screen, BLUE, (x + 60, y + 80), 8)
        else:
            offset = 0 if self.sim.is_sleeping else int(np.sin(pygame.time.get_ticks() / 200.0) * 10)
            pygame.draw.circle(self.screen, color, (x, y - 80 + offset), 20)
            pygame.draw.line(self.screen, color, (x, y - 60 + offset), (x, y + 20 + offset), 5)
            pygame.draw.line(self.screen, color, (x, y - 40 + offset), (x - 30, y - 10 + offset), 5)
            pygame.draw.line(self.screen, color, (x, y - 40 + offset), (x + 30, y - 10 + offset), 5)

            if self.sim.wearing_warm:
                pygame.draw.rect(self.screen, ORANGE, (x - 15, y - 60 + offset, 30, 60))
            if self.sim.wearing_mask:
                pygame.draw.circle(self.screen, BLUE, (x + 12, y - 80 + offset), 8)

    def draw_environment_ui(self):
        bg_color = BG_NORMAL
        env_name = "IDEAL ENVIRONMENT"

        if self.sim.target_env_temp <= 5.0:
            bg_color = BG_SNOW
            env_name = "SNOW MOUNTAIN"
        elif self.sim.target_aqi >= 150.0:
            bg_color = BG_CAVE
            env_name = "HAZARDOUS CAVE"

        self.screen.fill(bg_color)

        bg_text = large_font.render(env_name, True, (200, 200, 200))
        self.screen.blit(bg_text, (WIDTH // 2 - bg_text.get_width() // 2, HEIGHT // 2 - 150))

        pygame.draw.rect(self.screen, BLUE, (WIDTH - 270, 20, 260, 180), 2)
        self.screen.blit(font.render("ENVIRONMENT PARAMETERS", True, BLUE), (WIDTH - 260, 25))

        params = [
            ('temp', f"Temp: {self.sim.env_temp:.1f} °C (T: {self.sim.target_env_temp})"),
            ('oxy', f"Oxy: {self.sim.oxygen:.1f} % (T: {self.sim.target_oxygen})"),
            ('aqi', f"AQI: {self.sim.aqi:.0f} (T: {self.sim.target_aqi})")
        ]

        for key, text in params:
            rect = self.input_rects[key]
            if self.is_paused and self.active_input == key:
                pygame.draw.rect(self.screen, YELLOW, rect)
                self.screen.blit(font.render(self.input_text + "_", True, BLACK), (rect.x + 5, rect.y))
            else:
                if self.is_paused: pygame.draw.rect(self.screen, GRAY, rect, 1)
                self.screen.blit(font.render(text, True, BLACK), (rect.x + 5, rect.y))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if self.state == "PLAYING":
                    if not self.is_paused:
                        self.handle_mouse_swipe(event)
                    else:
                        self.handle_custom_inputs(event)

            if self.state == "MENU":
                self.screen.fill(WHITE)
                title = large_font.render("SIMULATION: AI SURVIVAL PREDICTOR", True, BLACK)
                self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

                # Chỉnh Base HR
                self.draw_button("-", 300, 280, 50, 40, GRAY,
                                 lambda: setattr(self.sim, 'base_hr', max(50, self.sim.base_hr - 5)))
                hr_text = font.render(f"Base heart rate: {self.sim.base_hr} bpm", True, BLACK)
                self.screen.blit(hr_text, (380, 290))
                self.draw_button("+", 650, 280, 50, 40, GRAY,
                                 lambda: setattr(self.sim, 'base_hr', min(100, self.sim.base_hr + 5)))

                # Chỉnh Base HRV
                self.draw_button("-", 300, 340, 50, 40, GRAY,
                                 lambda: setattr(self.sim, 'base_hrv', max(20, self.sim.base_hrv - 5)))
                hrv_text = font.render(f"Base HRV (Stress): {self.sim.base_hrv} ms", True, BLACK)
                self.screen.blit(hrv_text, (380, 350))
                self.draw_button("+", 650, 340, 50, 40, GRAY,
                                 lambda: setattr(self.sim, 'base_hrv', min(100, self.sim.base_hrv + 5)))

                self.draw_button("PLAY", WIDTH // 2 - 100, 500, 200, 60, GREEN, self.start_game)

            elif self.state == "PLAYING":
                current_time = pygame.time.get_ticks()

                if current_time - self.last_tick > TICK_RATE and not self.is_paused:
                    self.sim.update_environment()
                    self.sim.update_body()

                    # Cập nhật trạng thái từ mô hình AI Lai
                    self.sim.ai_status = self.sim.predict_danger()

                    # XỬ LÝ ĐẾM NGƯỢC CHẾT (Đã hoạt động chuẩn xác)
                    if self.sim.ai_status == 2:
                        self.sim.danger_timer += 1
                        if self.sim.danger_timer >= 10:
                            self.state = "GAMEOVER"
                    else:
                        self.sim.danger_timer = max(0, self.sim.danger_timer - 1)

                    self.last_tick = current_time

                # 1. Vẽ Môi trường
                self.draw_environment_ui()

                # 2. Khung Sinh lý (Hiển thị đầy đủ thông số cơ thể)
                pygame.draw.rect(self.screen, GREEN, (20, 20, 260, 210), 2)
                self.screen.blit(font.render("BODY PARAMETERS", True, GREEN), (30, 25))
                self.screen.blit(font.render(f"Heart rate: {self.sim.hr:.1f} bpm", True, BLACK), (30, 60))
                self.screen.blit(font.render(f"HRV (Stress): {self.sim.hrv:.1f} ms", True, BLACK), (30, 90))
                self.screen.blit(font.render(f"Body temp: {self.sim.body_temp:.1f} °C", True, BLACK), (30, 120))
                self.screen.blit(font.render(f"Acceleration: {self.sim.accel:.2f} G", True, BLACK), (30, 150))

                fall_status = "0" if self.sim.fall_duration == 0 else f"{self.sim.fall_duration} s"
                self.screen.blit(font.render(f"Fall duration: {fall_status}", True, BLACK), (30, 180))

                # 3. Hình nhân
                self.draw_player()
                if self.sim.is_fallen and self.sim.auto_stand_timer < 0:
                    self.draw_button("STAND UP", 450, 450, 100, 40, GRAY, lambda: setattr(self.sim, 'is_fallen', False))

                # 4. Vẽ nút tương tác
                self.draw_button("Snow", 20, 600, 100, 40, BLUE, self.set_preset_snow)
                self.draw_button("Deep cave", 130, 600, 100, 40, DARK_GRAY, self.set_preset_cave)
                self.draw_button("Normal", 240, 600, 100, 40, GREEN, self.set_preset_normal)

                self.draw_button("Clothes", 600, 600, 130, 40, ORANGE, self.toggle_clothes)
                self.draw_button("Mask", 740, 600, 130, 40, BLUE, self.toggle_mask)
                self.draw_button("Heavy exercise", 600, 650, 130, 40, RED, self.toggle_exercise)
                self.draw_button("Sleep", 740, 650, 130, 40, GRAY, self.toggle_sleep)

                pause_color = ORANGE if self.is_paused and (current_time % 1000 < 500) else YELLOW
                self.draw_button("PAUSE" if not self.is_paused else "UNPAUSE", WIDTH // 2 - 60, 20, 120, 40,
                                 pause_color, self.toggle_pause)

                # 5. CẢNH BÁO UI (Hộp thông báo nhấp nháy/hiển thị dựa trên AI status)
                if self.sim.ai_status == 1:
                    alert_box = pygame.Rect(WIDTH // 2 - 200, 100, 400, 80)
                    pygame.draw.rect(self.screen, ORANGE, alert_box)
                    self.screen.blit(large_font.render("CAUTION!", True, WHITE), (alert_box.x + 10, alert_box.y + 10))
                    self.screen.blit(font.render("Caution: AI detects hidden physiological risks.", True, WHITE),
                                     (alert_box.x + 10, alert_box.y + 50))

                elif self.sim.ai_status == 2:
                    alert_box = pygame.Rect(WIDTH // 2 - 250, 100, 500, 80)
                    pygame.draw.rect(self.screen, RED, alert_box)
                    self.screen.blit(large_font.render("WARNING! CRITICAL DANGER!", True, WHITE),
                                     (alert_box.x + 10, alert_box.y + 10))
                    self.screen.blit(
                        font.render(f"Critical health failure! Death imminent in: {10 - self.sim.danger_timer}s", True,
                                    WHITE),
                        (alert_box.x + 10, alert_box.y + 50))

            elif self.state == "GAMEOVER":
                self.screen.fill(BLACK)
                game_over_text = large_font.render("GAME OVER", True, RED)
                self.screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - 50))
                self.draw_button("REPLAY", WIDTH // 2 - 75, HEIGHT // 2 + 50, 150, 50, RED, self.restart_game)

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    app = GameApp()
    app.run()