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
DARK_GRAY = (50, 50, 50)
BLUE = (52, 152, 219)  # Môi trường
RED = (231, 76, 60)  # Nguy kịch
ORANGE = (243, 156, 18)  # Cảnh báo
GREEN = (46, 204, 113)  # Sức khỏe cơ thể
YELLOW = (241, 196, 15)

font = pygame.font.SysFont('Arial', 20)
large_font = pygame.font.SysFont('Arial', 32, bold=True)

# Load AI Model (Thay tên file pkl của bạn vào đây)
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
        # Baseline người chơi (Thay đổi từ Menu)
        self.age = 25
        self.base_hr = 75.0

        # Môi trường Target (Để tạo sự thay đổi dần dần)
        self.target_env_temp = 24.0
        self.target_oxygen = 20.9
        self.target_pressure = 1013.0
        self.target_aqi = 30.0

        # Môi trường hiện tại
        self.env_temp = 24.0
        self.oxygen = 20.9
        self.pressure = 1013.0
        self.aqi = 30.0
        self.humidity = 55.0
        self.uv = 2.0

        # Cơ thể hiện tại
        self.hr = self.base_hr
        self.hrv = 60.0
        self.body_temp = 36.8
        self.accel = 1.0
        self.fall_duration = 0

        # Bộ nhớ để tính Rolling STD (Sliding window)
        self.hr_history = deque(maxlen=12)

        # Tương tác (Nhiễu / Xử lý)
        self.is_sleeping = False
        self.is_heavy_exercise = False
        self.wearing_warm = False
        self.wearing_mask = False
        self.is_fallen = False

        # Trạng thái game
        self.ai_status = 0  # 0: An toàn, 1: Cảnh báo, 2: Nguy hiểm
        self.danger_timer = 0  # Đếm ngược tử vong

    def update_environment(self):
        # Môi trường thay đổi từ từ tiến về target
        self.env_temp += (self.target_env_temp - self.env_temp) * 0.1
        self.oxygen += (self.target_oxygen - self.oxygen) * 0.1
        self.pressure += (self.target_pressure - self.pressure) * 0.1
        self.aqi += (self.target_aqi - self.aqi) * 0.1

    def update_body(self):
        # Tính gia tốc
        if self.is_fallen:
            self.accel = 1.0
            self.fall_duration += 5
        else:
            self.fall_duration = 0
            if self.is_heavy_exercise:
                self.accel = random.uniform(1.2, 1.8)
            elif self.is_sleeping:
                self.accel = random.uniform(0.9, 1.0)
            else:
                self.accel = random.uniform(0.9, 1.2)  # Đi bộ bình thường

        # Tính nhịp tim (hr)
        target_hr = self.base_hr
        if self.is_heavy_exercise: target_hr += 60
        if self.is_sleeping: target_hr -= 15
        if self.oxygen < 19 and not self.wearing_mask: target_hr += 15  # Bù oxy

        self.hr += (target_hr - self.hr) * 0.2 + random.uniform(-1, 1)
        self.hr_history.append(self.hr)

        # Tính thân nhiệt
        temp_loss = (24.0 - self.env_temp) * 0.005
        if self.wearing_warm: temp_loss *= 0.2  # Giảm mất nhiệt nếu mặc ấm
        if self.is_heavy_exercise: temp_loss -= 0.05  # Vận động sinh nhiệt

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

    def predict_danger(self):
        df = self.get_features_for_ai()
        if rf_model is not None:
            return rf_model.predict(df)[0]
        else:
            # Mô phỏng Rule-based nếu không có model
            if self.fall_duration > 0 and self.body_temp < 35: return 2
            if self.env_temp > 38 and self.body_temp > 38: return 2
            if self.body_temp < 35.5 or self.oxygen < 18 or (self.accel > 4 and self.fall_duration == 0): return 1
            return 0


# ==========================================
# 3. GIAO DIỆN VÀ MAIN GAME
# ==========================================
class GameApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Explorer Simulator")
        self.clock = pygame.time.Clock()
        self.state = "MENU"
        self.sim = SimulationData()
        self.last_tick = pygame.time.get_ticks()
        self.is_paused = False

        # Mouse swipe detection
        self.mouse_down_pos = None

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
            pygame.time.delay(150)  # Debounce
            action()

    def set_preset_snow(self):
        self.sim.target_env_temp = -15.0
        self.sim.target_oxygen = 19.0

    def set_preset_cave(self):
        self.sim.target_aqi = 250.0
        self.sim.target_oxygen = 17.0

    def set_preset_normal(self):
        self.sim.target_env_temp = 24.0
        self.sim.target_oxygen = 20.9
        self.sim.target_aqi = 30.0

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

    def start_game(self):
        self.state = "PLAYING"

    def restart_game(self):
        self.sim = SimulationData()
        self.state = "PLAYING"

    def handle_mouse_swipe(self, event):
        # Tính toán để hất ngã hình nhân
        player_rect = pygame.Rect(450, 300, 100, 200)  # Khung hình nhân
        if event.type == pygame.MOUSEBUTTONDOWN:
            if player_rect.collidepoint(event.pos):
                self.mouse_down_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.mouse_down_pos:
                dx = event.pos[0] - self.mouse_down_pos[0]
                dy = event.pos[1] - self.mouse_down_pos[1]
                speed = np.sqrt(dx ** 2 + dy ** 2)
                if speed > 100:  # Lướt chuột nhanh
                    self.sim.accel = speed / 50.0  # Gia tốc cực lớn tạm thời
                    self.sim.is_fallen = True
                    print(f"Felt! Acceleration: {self.sim.accel:.2f}G")
                self.mouse_down_pos = None

    def draw_player(self):
        # Vẽ hình nhân (Stickman) đơn giản
        color = BLACK if not self.sim.is_fallen else RED
        x, y = 500, 400

        if self.sim.is_fallen:
            pygame.draw.circle(self.screen, color, (x + 50, y + 80), 20)  # Đầu nằm ngang
            pygame.draw.line(self.screen, color, (x + 30, y + 80), (x - 40, y + 80), 5)  # Thân
        else:
            offset = 0 if self.sim.is_sleeping else int(np.sin(pygame.time.get_ticks() / 200.0) * 10)
            pygame.draw.circle(self.screen, color, (x, y - 80 + offset), 20)  # Đầu
            pygame.draw.line(self.screen, color, (x, y - 60 + offset), (x, y + 20 + offset), 5)  # Thân
            pygame.draw.line(self.screen, color, (x, y - 40 + offset), (x - 30, y - 10 + offset), 5)  # Tay trái
            pygame.draw.line(self.screen, color, (x, y - 40 + offset), (x + 30, y - 10 + offset), 5)  # Tay phải

        # Hiển thị trang bị
        if self.sim.wearing_warm:
            pygame.draw.rect(self.screen, ORANGE, (x - 15, y - 60 + offset, 30, 60))
        if self.sim.wearing_mask:
            pygame.draw.circle(self.screen, BLUE, (x + 10, y - 80 + offset), 8)

    def run(self):
        while True:
            self.screen.fill(WHITE)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if self.state == "PLAYING" and not self.is_paused:
                    self.handle_mouse_swipe(event)

            if self.state == "MENU":
                # Màn hình Menu đơn giản
                title = large_font.render("SIMULATION", True, BLACK)
                self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

                # Nút điều chỉnh Base HR
                self.draw_button("-", 350, 300, 50, 40, GRAY,
                                 lambda: setattr(self.sim, 'base_hr', max(50, self.sim.base_hr - 5)))
                hr_text = font.render(f"Base heart rate: {self.sim.base_hr} bpm", True, BLACK)
                self.screen.blit(hr_text, (420, 310))
                self.draw_button("+", 650, 300, 50, 40, GRAY,
                                 lambda: setattr(self.sim, 'base_hr', min(100, self.sim.base_hr + 5)))

                self.draw_button("PLAY", WIDTH // 2 - 100, 500, 200, 60, GREEN, self.start_game)

            elif self.state == "PLAYING":
                current_time = pygame.time.get_ticks()

                # --- LOGIC TICK (Cập nhật dữ liệu mỗi giây) ---
                if current_time - self.last_tick > TICK_RATE and not self.is_paused:
                    self.sim.update_environment()
                    self.sim.update_body()

                    # AI Dự đoán
                    self.sim.ai_status = self.sim.predict_danger()

                    # Xử lý tử vong
                    if self.sim.ai_status == 2:
                        self.sim.danger_timer += 1
                        if self.sim.danger_timer >= 10:  # Nguy hiểm quá 10 tick
                            self.state = "GAMEOVER"
                    else:
                        self.sim.danger_timer = max(0, self.sim.danger_timer - 1)

                    self.last_tick = current_time

                # --- VẼ GIAO DIỆN UI ---
                # Khung Sinh lý (Xanh lá)
                pygame.draw.rect(self.screen, GREEN, (20, 20, 250, 180), 2)
                self.screen.blit(font.render("BODY PARAMETERS", True, GREEN), (30, 25))
                self.screen.blit(font.render(f"Heart rate: {self.sim.hr:.1f} bpm", True, BLACK), (30, 60))
                self.screen.blit(font.render(f"Body temp: {self.sim.body_temp:.1f} °C", True, BLACK), (30, 90))
                self.screen.blit(font.render(f"Acceleration: {self.sim.accel:.2f} G", True, BLACK), (30, 120))
                self.screen.blit(font.render(f"Fall duration: {self.sim.fall_duration} s", True, BLACK), (30, 150))

                # Khung Môi trường (Xanh dương)
                pygame.draw.rect(self.screen, BLUE, (WIDTH - 270, 20, 250, 180), 2)
                self.screen.blit(font.render("ENVIRONMENT PARAMETERS", True, BLUE), (WIDTH - 260, 25))
                self.screen.blit(font.render(f"Environment temp: {self.sim.env_temp:.1f} °C", True, BLACK), (WIDTH - 260, 60))
                self.screen.blit(font.render(f"Oxy: {self.sim.oxygen:.1f} %", True, BLACK), (WIDTH - 260, 90))
                self.screen.blit(font.render(f"AQI: {self.sim.aqi:.0f}", True, BLACK), (WIDTH - 260, 120))

                # Hình nhân
                self.draw_player()
                if self.sim.is_fallen:
                    self.draw_button("STAND UP", 450, 450, 100, 40, GRAY, lambda: setattr(self.sim, 'is_fallen', False))

                # Vẽ nút Tương tác & Môi trường
                self.draw_button("Snow", 20, 600, 100, 40, BLUE, self.set_preset_snow)
                self.draw_button("Deep cave", 130, 600, 100, 40, DARK_GRAY, self.set_preset_cave)
                self.draw_button("Normal", 240, 600, 100, 40, GREEN, self.set_preset_normal)

                self.draw_button("Clothes", 600, 600, 130, 40, ORANGE, self.toggle_clothes)
                self.draw_button("Mask", 740, 600, 130, 40, BLUE, self.toggle_mask)
                self.draw_button("Heavy exercise", 600, 650, 130, 40, RED, self.toggle_exercise)
                self.draw_button("Sleep", 740, 650, 130, 40, GRAY, self.toggle_sleep)

                self.draw_button("PAUSE" if not self.is_paused else "CONTINUE", WIDTH // 2 - 60, 20, 120, 40, YELLOW,
                                 self.toggle_pause)

                # --- HỆ THỐNG CẢNH BÁO AI ---
                if self.sim.ai_status == 1:
                    alert_box = pygame.Rect(WIDTH // 2 - 200, 100, 400, 80)
                    pygame.draw.rect(self.screen, ORANGE, alert_box)
                    self.screen.blit(large_font.render("CAUTION!", True, WHITE),
                                     (alert_box.x + 10, alert_box.y + 10))
                    self.screen.blit(font.render("Caution", True, WHITE),
                                     (alert_box.x + 10, alert_box.y + 50))

                elif self.sim.ai_status == 2:
                    alert_box = pygame.Rect(WIDTH // 2 - 250, 100, 500, 80)
                    pygame.draw.rect(self.screen, RED, alert_box)
                    self.screen.blit(large_font.render("WARNING!", True, WHITE),
                                     (alert_box.x + 10, alert_box.y + 10))

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