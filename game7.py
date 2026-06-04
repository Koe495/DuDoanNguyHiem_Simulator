# game6.py
"""
Game Mô phỏng Dự đoán Nguy hiểm – v6
======================================
Điều khiển:
  SPACE / P    : Tạm dừng / Tiếp tục
  ESC          : Về menu
  Lướt chuột nhanh qua hình nhân : Gây ngã (nhanh = ngã mạnh)
  Click thông số môi trường (khi dừng) : Chỉnh sửa
  Các nút dưới màn hình : Toggle trạng thái người chơi

Yêu cầu:
  pip install pygame numpy joblib scikit-learn
  Cùng thư mục: rf_model_v8.pkl, feature_names_v8.json
"""

import sys
import math
import random
from collections import deque

import pygame
import numpy as np
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
FPS    = 30
SW, SH = 1280, 720
GAME_W = 800
STAT_W = SW - GAME_W          # 480
TOP_H  = 55
BOT_H  = 125
MAIN_H = SH - TOP_H - BOT_H  # 540

WIN_SZ       = 50             # ML window (50 × 0.2s = 10s — khớp train7.py WINDOW_SIZE=50)
SAMPLE_EVERY = 6              # frames between samples (0.2 s)
CRIT_DEATH   = FPS * 12       # frames at label-2 before death

DT   = 1.0 / FPS              # seconds per frame
# Scale factors mapping gendata5 per-5s coefficients to per-frame
HR_K   = 1.0 - (0.88 ** (DT / 5.0))   # ≈ 0.000856
HRV_K  = 1.0 - (0.92 ** (DT / 5.0))   # ≈ 0.000556
TEMP_K = DT / 5.0                       # 0.00667

SENSOR_COLS = [
    "hr", "hrv", "body_temp", "env_temp", "humidity",
    "pressure", "oxygen", "uv_index", "aqi",
    "accel_mag", "fall_duration",
]

PRESETS: dict = {
    "Ly tuong":  dict(env_temp=22,  humidity=55,  pressure=1010, oxygen=20.9, uv_index=2,  aqi=25 ),
    "Nui tuyet": dict(env_temp=-18, humidity=30,  pressure=750,  oxygen=18.0, uv_index=4,  aqi=10 ),
    "Sa mac":    dict(env_temp=42,  humidity=10,  pressure=1005, oxygen=20.9, uv_index=9,  aqi=35 ),
    "Hang sau":  dict(env_temp=15,  humidity=90,  pressure=1020, oxygen=15.5, uv_index=0,  aqi=180),
    "Chay rung": dict(env_temp=38,  humidity=20,  pressure=1000, oxygen=15.0, uv_index=6,  aqi=420),
    "Bac cuc":   dict(env_temp=-32, humidity=20,  pressure=975,  oxygen=20.9, uv_index=1,  aqi=8  ),
}
PRESET_KEYS = list(PRESETS.keys())

PRESET_LABELS = {
    "Ly tuong":  "Ly tuong",
    "Nui tuyet": "Nui tuyet",
    "Sa mac":    "Sa mac",
    "Hang sau":  "Hang sau",
    "Chay rung": "Chay rung",
    "Bac cuc":   "Bac cuc",
}

TERRAIN_COL: dict = {
    "Ly tuong":  {"sky_t": (95, 155, 225),  "sky_b": (155, 200, 245), "gnd": (55, 120, 55)},
    "Nui tuyet": {"sky_t": (75, 95,  140),  "sky_b": (155, 175, 205), "gnd": (215, 225, 235)},
    "Sa mac":    {"sky_t": (205, 138, 55),  "sky_b": (235, 195, 115), "gnd": (185, 158, 88)},
    "Hang sau":  {"sky_t": (8,   8,  14),   "sky_b": (22,  22,  38),  "gnd": (38,  32,  26)},
    "Chay rung": {"sky_t": (145, 55,  18),  "sky_b": (195, 95,  35),  "gnd": (48,  32,  18)},
    "Bac cuc":   {"sky_t": (58,  78, 118),  "sky_b": (128, 165, 205), "gnd": (205, 220, 235)},
}

PARAM_RANGE: dict = {
    "env_temp": (-50.0, 55.0), "humidity": (0.0, 100.0), "pressure": (600.0, 1050.0),
    "oxygen":   (10.0,  21.0), "uv_index": (0.0,  12.0), "aqi":      (0.0,   600.0),
}

LABEL_NAME = ["AN TOAN", "CANH BAO", "NGUY KICH"]
LABEL_COL  = [(55, 195, 75), (225, 175, 25), (215, 55, 55)]
PRED_BG    = [(14, 38, 20), (48, 40, 8), (48, 10, 10)]
PRED_BD    = [(38, 175, 65), (215, 162, 15), (195, 35, 35)]

# ─── Color palette ─────────────────────────────────────────────────────────
BG      = (14, 19, 33)
TOPBG   = (19, 26, 45)
BOTBG   = (19, 26, 45)
PHY_BG  = (14, 28, 20); PHY_BD = (38, 155, 75)
ENV_BG  = (14, 20, 38); ENV_BD = (55, 115, 195)
TXT     = (215, 225, 242)
DIM     = (115, 132, 158)
WHITE   = (255, 255, 255)
GREEN   = (55, 195, 75)
YELLOW  = (225, 175, 25)
RED     = (215, 55, 55)
ORANGE  = (225, 125, 28)
BTN_N   = (38, 52, 85)
BTN_H   = (62, 82, 128)
BTN_A   = (28, 95, 55)
BTN_AH  = (42, 125, 75)
INP_BG  = (22, 32, 58); INP_BD = (95, 145, 215)
CHAR_B  = (75, 135, 215)
CHAR_S  = (252, 205, 170)

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rp(px: float, py: float, cx: float, cy: float, a: float) -> tuple:
    """Rotate point (px,py) around (cx,cy) by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    dx, dy = px - cx, py - cy
    return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)


def lerp_col(c1: tuple, c2: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_rrect(surf: pygame.Surface, col: tuple, rect: tuple,
               r: int = 7, w: int = 0) -> None:
    pygame.draw.rect(surf, col, rect, w, border_radius=r)


def draw_panel(surf: pygame.Surface, rect: tuple, bg: tuple,
               bdr: tuple, title: str = "", f=None, bdr_w: int = 2) -> None:
    draw_rrect(surf, bg, rect)
    draw_rrect(surf, bdr, rect, w=bdr_w)
    if title and f:
        surf.blit(f.render(title, True, bdr), (rect[0] + 9, rect[1] + 6))


def zcr(sig: np.ndarray) -> float:
    c = sig - float(np.mean(sig))
    return float(np.sum(np.diff(np.sign(c)) != 0)) / max(len(sig) - 1, 1)


def extract_features(buffer: deque) -> np.ndarray:
    """Extract 44 time-domain features from a WIN_SZ × 11 rolling buffer."""
    arr = np.array(buffer, dtype=float)
    feats = []
    for j in range(arr.shape[1]):
        s = arr[:, j]
        feats += [float(np.mean(s)), float(np.std(s)),
                  float(np.sqrt(np.mean(s ** 2))), zcr(s)]
    return np.array(feats, dtype=float)


def wrap_lines(text: str, font: pygame.font.Font, max_w: int) -> list:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.size(t)[0] <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def get_suggestion(label: int, hr: float, hrv: float, body_temp: float,
                   oxygen: float, aqi: float, trauma: float,
                   dehydration: float = 0.0) -> str:
    if label == 0:
        if dehydration > 0.3:
            return "Bat dau mat nuoc - Hay uong nuoc som."
        return "Tat ca chi so on dinh. Tiep tuc hanh trinh an toan."
    if label == 1:
        if body_temp < 35.5:
            return "Than nhiet giam - Mac them quan ao am ngay!"
        if body_temp > 38.5:
            return "Than nhiet tang - Nghi ngoi, tim bong mat, uong nuoc."
        if oxygen < 18.0:
            return "Oxy thap - Deo mat na duong khi."
        if aqi > 150:
            return "Khong khi o nhiem - Deo mat na bao ho."
        if dehydration > 0.5:
            return "Mat nuoc nghiem trong - Uong nuoc ngay, nghi ngoi!"
        if hr > 140:
            return "Nhip tim cao - Giam cuong do van dong, nghi ngoi."
        if hrv < 25:
            return "HRV thap - Co the cang thang. Nghi ngoi ngay."
        if trauma > 0.2:
            return "Phat hien chan thuong - Tranh van dong manh."
        return "Nguy co tiem an - Theo doi ky cac chi so."
    # label == 2
    if body_temp < 34.0:
        return "HA THAN NHIET NGUY KICH! Tim noi am ap, mac du quan ao am!"
    if body_temp > 40.5:
        return "SOC NHIET NGUY KICH! Lam mat co the ngay, khong van dong!"
    if oxygen < 15.0:
        return "THIEU OXY NGUY KICH! Deo mat na ngay, roi khoi khu vuc!"
    if aqi > 350:
        return "NGO DOC KHI! Deo mat na va thoat khoi day ngay!"
    if dehydration > 0.8:
        return "KIET NUOC NGUY KICH! Uong nuoc va cap cuu ngay!"
    if trauma > 0.6:
        return "CHAN THUONG NANG! Khong di chuyen, goi cuu ho ngay!"
    if hr > 175 or hr < 42:
        return "TIM BAT THUONG! Nam xuong, goi cuu ho ngay!"
    return "NGUY HIEM TINH MANG! Can cap cuu ngay lap tuc!"


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSIO SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PhysioSim:
    """Per-frame physiological simulation matching gendata5/6 equations."""

    def __init__(self, base_hr: float, base_hrv: float) -> None:
        self.base_hr   = float(base_hr)
        self.base_hrv  = float(base_hrv)

        self.hr        = float(base_hr)
        self.hrv       = float(base_hrv)
        self.body_temp = 36.8

        # Weather (current → target smoothed)
        self.env_temp   = 22.0;  self.t_env_temp  = 22.0
        self.humidity   = 55.0;  self.t_humidity  = 55.0
        self.pressure   = 1010.0;self.t_pressure  = 1010.0
        self.oxygen     = 20.9;  self.t_oxygen    = 20.9
        self.uv_index   = 2.0;   self.t_uv_index  = 2.0
        self.aqi        = 25.0;  self.t_aqi       = 25.0

        self.terrain    = "Ly tuong"

        # Player state
        self.clothing    = 0.5   # 0=naked, 1=full winter gear
        self.trauma      = 0.0
        self.dehydration = 0.0   # 0=hydrated, 1=severely dehydrated
        self.mask        = False  # oxygen mask
        self.sleeping    = False
        self.exercising  = False
        self.fallen      = False
        self.fall_timer  = 0.0   # seconds since falling
        self.fall_dur    = 0.0   # seconds total (for feature)
        self.accel_mag   = 1.0
        self.walk_phase  = 0.0   # radians, for animation

        self.buffer      = deque(maxlen=WIN_SZ)
        self.sample_cnt  = 0
        self.label       = 0
        self.confidence  = 1.0
        self.crit_frames = 0

    # ── public controls ─────────────────────────────────────────

    def apply_preset(self, key: str) -> None:
        self.terrain = key
        p = PRESETS[key]
        self.t_env_temp = float(p["env_temp"])
        self.t_humidity = float(p["humidity"])
        self.t_pressure = float(p["pressure"])
        self.t_oxygen   = float(p["oxygen"])
        self.t_uv_index = float(p["uv_index"])
        self.t_aqi      = float(p["aqi"])

    def do_knock(self, intensity: float) -> None:
        """intensity 0..1 ; triggers fall + trauma."""
        self.accel_mag   = 1.0 + intensity * 8.0
        self.trauma      = min(1.0, self.trauma + intensity * 0.35)
        self.fallen      = True
        self.fall_timer  = 0.0
        self.sleeping    = False
        self.exercising  = False

    def add_clothing(self) -> None:
        self.clothing = min(1.0, self.clothing + 0.18)

    def remove_clothing(self) -> None:
        self.clothing = max(0.0, self.clothing - 0.18)

    def stand_up(self) -> None:
        self.fallen     = False
        self.fall_timer = 0.0
        self.fall_dur   = 0.0

    def drink(self) -> None:
        """Giảm dehydration khi uống nước (~0.15 mỗi lần)."""
        self.dehydration = max(0.0, self.dehydration - 0.15)

    # ── per-frame update ────────────────────────────────────────

    def update(self) -> None:
        alpha = 0.018  # weather transition smoothness

        # Smooth weather toward targets
        self.env_temp += (self.t_env_temp - self.env_temp) * alpha
        self.humidity += (self.t_humidity - self.humidity) * alpha
        self.pressure += (self.t_pressure - self.pressure) * alpha
        self.oxygen   += (self.t_oxygen   - self.oxygen  ) * alpha
        self.uv_index += (self.t_uv_index - self.uv_index) * alpha
        self.aqi      += (self.t_aqi      - self.aqi     ) * alpha

        # Exertion
        if self.fallen:
            exertion = -0.25
        elif self.sleeping:
            exertion = -0.15
        elif self.exercising:
            exertion = 0.78
        else:
            exertion = 0.18

        # Walk phase (animation)
        speed = max(0.0, exertion + 0.3)
        self.walk_phase = (self.walk_phase + speed * 0.12) % (2 * math.pi)

        # Effective oxygen / aqi (mask effect)
        eff_o2  = min(20.9, self.oxygen + (3.8 if self.mask else 0.0))
        eff_aqi = max(0, self.aqi * (0.15 if self.mask else 1.0))

        # Fall management
        if self.fallen:
            self.fall_timer += DT
            self.fall_dur    = self.fall_timer
            if self.fall_timer > 8.0:   # auto-recover after 8 s
                self.fallen     = False
                self.fall_timer = 0.0
                self.fall_dur   = 0.0
            self.trauma = min(1.0, self.trauma + 0.015 * TEMP_K * 150)
        else:
            self.fall_dur = 0.0
            self.trauma   = max(0.0, self.trauma - 0.0008)

        # Accel: settle to baseline when upright
        if not self.fallen:
            tgt_accel = 1.0 + max(0, exertion) * 0.12
            self.accel_mag += (tgt_accel - self.accel_mag) * 0.08
        else:
            self.accel_mag += (0.1 - self.accel_mag) * 0.04  # near-zero in freefall

        # Dehydration: accumulates with exertion + heat; reduced by drinking
        dehy_rate = (0.0003
                     + max(0.0, exertion) * 0.0004
                     + max(0.0, self.env_temp - 28.0) * 0.00005)
        self.dehydration = float(np.clip(self.dehydration + dehy_rate, 0.0, 1.0))

        # A. Body temperature  (heat_prod – heat_xfer scaled to per-frame)
        hp = 0.012 * (1.0 + exertion) * TEMP_K
        hl = 0.004 * (self.env_temp - self.body_temp) * (1.0 - self.clothing) * TEMP_K
        dh = self.dehydration * 0.005 * TEMP_K
        self.body_temp += hp + hl + dh + random.gauss(0, 0.006)
        self.body_temp  = float(np.clip(self.body_temp, 30.0, 43.0))

        # B. Heart rate
        hyp  = max(0.0, 20.9 - eff_o2) * 4.5
        tox  = max(0.0, eff_aqi - 100.0) * 0.04
        shk  = self.trauma * 40.0 if exertion >= 0 else -self.trauma * 25.0
        dhy  = self.dehydration * 20.0
        t_hr = self.base_hr + exertion * 65.0 + hyp + tox + shk + dhy
        self.hr += (t_hr - self.hr) * HR_K * 150 + random.gauss(0, 0.4)
        self.hr  = float(np.clip(self.hr, 32.0, 198.0))

        # C. HRV
        th_s  = abs(self.body_temp - 36.8) * 16.0
        hy_s  = max(0.0, 20.9 - eff_o2) * 10.0
        tx_s  = max(0.0, eff_aqi - 50.0) * 0.12
        inj_s = self.trauma * 45.0
        dhy_s = self.dehydration * 15.0
        t_hrv = self.base_hrv - exertion * 20.0 - th_s - hy_s - tx_s - inj_s - dhy_s
        self.hrv += (t_hrv - self.hrv) * HRV_K * 150 + random.gauss(0, 0.3)
        self.hrv  = float(np.clip(self.hrv, 4.0, 100.0))

        # Sample for ML buffer
        self.sample_cnt += 1
        if self.sample_cnt >= SAMPLE_EVERY:
            self.sample_cnt = 0
            # Dùng eff_o2 / eff_aqi (đã tính hiệu quả mặt nạ) thay vì raw
            # để ML model nhận đúng tín hiệu "cơ thể đang trải nghiệm gì"
            self.buffer.append([
                round(self.hr, 1), round(self.hrv, 1), round(self.body_temp, 2),
                round(self.env_temp, 1), round(self.humidity, 1),
                round(self.pressure, 1), round(eff_o2, 2),
                round(self.uv_index, 1), int(eff_aqi),
                round(self.accel_mag, 2), round(self.fall_dur, 1),
            ])

    def get_reading(self) -> list:
        return [
            self.hr, self.hrv, self.body_temp, self.env_temp,
            self.humidity, self.pressure, self.oxygen,
            self.uv_index, self.aqi, self.accel_mag, self.fall_dur,
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  ML PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MLPredictor:
    """Loads rf_model_v8.pkl and predicts risk label from WIN_SZ×0.2s rolling window."""

    def __init__(self) -> None:
        self.model = None
        self.ok    = False
        self._load()

    def _load(self) -> None:
        if not _HAS_JOBLIB:
            return
        try:
            self.model = joblib.load("rf_model_v8.pkl")
            self.ok = True
            print("[MLPredictor] Model loaded OK.")
        except FileNotFoundError as e:
            print(f"[MLPredictor] Warning: {e} — using rule-based fallback.")
        except Exception as e:
            print(f"[MLPredictor] Error loading model: {e}")

    def predict(self, buffer: deque,
               base_hr: float = 70.0, base_hrv: float = 62.0) -> tuple:
        """
        Trả về (label, confidence).
        base_hr, base_hrv: baseline cá nhân của người chơi – được thêm vào
        feature vector để ML model đưa ra dự đoán cá nhân hoá.
        Fallback về rule-based (cũng cá nhân hoá) nếu thiếu model.
        """
        if len(buffer) < WIN_SZ:
            return 0, 0.0

        arr = np.array(buffer, dtype=float)

        if self.ok:
            try:
                # 44 đặc trưng time-domain + 2 baseline = 46 features
                feats = np.append(
                    extract_features(buffer),
                    [base_hr, base_hrv],
                ).reshape(1, -1)
                proba = self.model.predict_proba(feats)[0]
                lbl   = int(np.argmax(proba))
                return lbl, float(proba[lbl])
            except Exception:
                pass

        # ── Rule-based fallback cá nhân hoá ──────────────────────────────────
        hr_m  = float(np.mean(arr[:, 0]))
        hrv_m = float(np.mean(arr[:, 1]))
        bt_m  = float(np.mean(arr[:, 2]))
        env_m = float(np.mean(arr[:, 3]))
        oxy_m = float(np.mean(arr[:, 6]))
        uv_m  = float(np.mean(arr[:, 7]))
        aqi_m = float(np.mean(arr[:, 8]))
        fd_m  = float(np.mean(arr[:, 10]))

        dev = 0.0

        # Môi trường / tuyệt đối
        dev += (abs(bt_m - 36.8) / 1.3) ** 2
        dev += (max(0.0, 20.9 - oxy_m) / 2.0) ** 2
        dev += (max(0.0, aqi_m - 120.0) / 100.0) ** 2
        if fd_m > 15:
            dev += fd_m / 25.0
        uv_heat = (max(0.0, uv_m - 5.0) / 7.0
                   + max(0.0, env_m - 32.0) / 12.0)
        dev += uv_heat * (1.0 + max(0.0, bt_m - 37.0) / 2.0)

        # HRV cá nhân hoá: cảnh báo khi < 65% baseline
        hrv_thresh = base_hrv * 0.65
        hrv_norm   = max(base_hrv * 0.25, 4.0)
        dev += (max(0.0, hrv_thresh - hrv_m) / hrv_norm) ** 2

        # HR cá nhân hoá: cảnh báo khi tăng > 55 BPM so với baseline
        dev += (max(0.0, hr_m - (base_hr + 55.0)) / 25.0) ** 2

        if dev >= 4.0:
            return 2, 0.85
        elif dev >= 1.2:
            return 1, 0.75
        return 0, 0.90


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARACTER ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

class Character:
    """Animated stick-figure drawn with pygame primitives."""

    # anchor = hip point
    def __init__(self, cx: int, cy: int) -> None:
        self.cx = cx
        self.cy = cy
        self.tilt   = 0.0    # current tilt angle (rad); 0=upright, π/2=lying
        self.t_tilt = 0.0    # target tilt
        self.blink  = 0      # blink timer
        self.zzz_phase = 0.0

    def update(self, sim: "PhysioSim", dead: bool) -> None:
        # Target tilt
        if dead:
            self.t_tilt = math.pi / 2
        elif sim.fallen or sim.sleeping:
            self.t_tilt = math.pi / 2
        else:
            self.t_tilt = 0.0

        # Smooth tilt
        diff = self.t_tilt - self.tilt
        self.tilt += diff * 0.12

        self.blink = (self.blink + 1) % 90
        if sim.sleeping or sim.fallen:
            self.zzz_phase = (self.zzz_phase + 0.04) % (2 * math.pi)

    def _pt(self, lx: float, ly: float, extra_angle: float = 0.0) -> tuple:
        """Rotate local point (lx, ly relative to hip) by tilt + extra."""
        a = self.tilt + extra_angle
        return rp(self.cx + lx, self.cy + ly, self.cx, self.cy, a)

    def draw(self, surf: pygame.Surface, sim: "PhysioSim",
             dead: bool, label: int) -> None:
        phase = sim.walk_phase
        exerc = sim.exercising
        amp   = 0.45 if exerc else 0.30     # leg swing amplitude
        body_col = CHAR_B if not dead else (155, 45, 45)
        skin_col = CHAR_S if not dead else (200, 100, 100)
        lw = 4

        # ── torso ──────────────────────────────────────────────────────
        hip   = self._pt(0, 0)
        neck  = self._pt(0, -55)
        pygame.draw.line(surf, body_col, hip, neck, lw)

        # ── head ───────────────────────────────────────────────────────
        head  = self._pt(0, -73)
        pygame.draw.circle(surf, skin_col, (int(head[0]), int(head[1])), 16)
        pygame.draw.circle(surf, body_col, (int(head[0]), int(head[1])), 16, 2)

        # Eyes
        er = 3
        eye_l = self._pt(-6, -77)
        eye_r = self._pt( 6, -77)
        if dead:
            # X eyes
            for ex, ey in [eye_l, eye_r]:
                ex, ey = int(ex), int(ey)
                pygame.draw.line(surf, RED, (ex-er, ey-er), (ex+er, ey+er), 2)
                pygame.draw.line(surf, RED, (ex+er, ey-er), (ex-er, ey+er), 2)
        elif sim.sleeping or sim.fallen:
            for ex, ey in [eye_l, eye_r]:
                pygame.draw.line(surf, body_col,
                                 (int(ex)-er, int(ey)), (int(ex)+er, int(ey)), 2)
        else:
            blink_open = self.blink < 80
            for ex, ey in [eye_l, eye_r]:
                pygame.draw.circle(surf, body_col, (int(ex), int(ey)),
                                   er if blink_open else 1)

        # ── arms ───────────────────────────────────────────────────────
        shldr = self._pt(0, -48)
        la    = -0.6 - math.sin(phase) * 0.5
        ra    =  0.6 + math.sin(phase) * 0.5
        arm_len = 36
        hand_l = self._pt(0, -48, la)
        hand_l = rp(shldr[0] + arm_len * math.sin(la),
                    shldr[1] + arm_len * math.cos(la),
                    shldr[0], shldr[1], 0)
        hand_r = (shldr[0] + arm_len * math.sin(ra),
                  shldr[1] + arm_len * math.cos(ra))
        # Recalculate hands relative to tilt
        def arm_end(side: int) -> tuple:
            ang = self.tilt + (side * amp * math.sin(phase))
            lx_arm = side * arm_len * math.sin(ang + side * 0.5)
            ly_arm = arm_len * math.cos(ang + side * 0.5) * 0.6
            base = self._pt(0, -48)
            dx = arm_len * math.sin(ang + side * math.pi * 0.18)
            dy = arm_len * math.cos(ang + side * math.pi * 0.18) * 0.55
            return (base[0] + dx, base[1] + dy)

        def leg_end(side: int) -> tuple:
            ang = self.tilt + (side * amp * math.sin(phase + math.pi))
            leg_len = 48
            base = self._pt(0, 0)
            dx = leg_len * math.sin(ang + side * math.pi * 0.09)
            dy = leg_len * math.cos(ang + side * math.pi * 0.09) * 0.75
            return (base[0] + dx, base[1] + dy)

        sh_pt = self._pt(0, -48)
        pygame.draw.line(surf, skin_col, sh_pt, arm_end(-1), lw)
        pygame.draw.line(surf, skin_col, sh_pt, arm_end( 1), lw)

        # ── legs ───────────────────────────────────────────────────────
        pygame.draw.line(surf, body_col, hip, leg_end(-1), lw + 1)
        pygame.draw.line(surf, body_col, hip, leg_end( 1), lw + 1)

        # ── ZZZ (sleep / fallen) ───────────────────────────────────────
        if (sim.sleeping or sim.fallen) and not dead:
            for i in range(3):
                t = (self.zzz_phase + i * 0.9) % (2 * math.pi)
                alpha_z = (math.sin(t) + 1) / 2
                sz = 10 + i * 4
                zx = int(head[0]) + 22 + i * 8
                zy = int(head[1]) - 15 - i * 14
                col_z = lerp_col(body_col, (200, 200, 200), alpha_z)
                # Small Z shape with lines
                zf = pygame.font.SysFont("arial", sz, bold=True)
                zs = zf.render("Z", True, col_z)
                surf.blit(zs, (zx, zy))

        # ── Clothing layers ────────────────────────────────────────────
        if sim.clothing > 0.6:
            # Draw a thick coat layer around torso
            tc = lerp_col(body_col, (30, 60, 140), (sim.clothing - 0.6) / 0.4)
            torso_r = pygame.Rect(int(hip[0]) - 12, int(neck[1]), 24, int(hip[1]) - int(neck[1]))
            pygame.draw.rect(surf, tc, torso_r, border_radius=6)

        # ── Oxygen mask ────────────────────────────────────────────────
        if sim.mask:
            mx, my = int(head[0]), int(head[1]) + 6
            pygame.draw.ellipse(surf, (100, 160, 100),
                                (mx - 12, my - 6, 24, 12))
            pygame.draw.ellipse(surf, (60, 200, 80),
                                (mx - 12, my - 6, 24, 12), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMPLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class Button:
    def __init__(self, rect: tuple, label: str, font: pygame.font.Font,
                 toggle: bool = False) -> None:
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.font   = font
        self.toggle = toggle
        self.active = False   # toggled on?
        self.hovered = False

    def handle(self, event: pygame.event.Event) -> bool:
        """Returns True on click."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.toggle:
                    self.active = not self.active
                return True
        return False

    def draw(self, surf: pygame.Surface) -> None:
        if self.active:
            col = BTN_AH if self.hovered else BTN_A
        else:
            col = BTN_H if self.hovered else BTN_N
        draw_rrect(surf, col, self.rect)
        draw_rrect(surf, (80, 100, 160) if not self.active else (60, 180, 90),
                   self.rect, w=2)
        txt = self.font.render(self.label, True, WHITE)
        surf.blit(txt, txt.get_rect(center=self.rect.center))


class TextInput:
    """Single-line numeric text input overlay."""

    def __init__(self, rect: tuple, font: pygame.font.Font,
                 init_val: str = "") -> None:
        self.rect  = pygame.Rect(rect)
        self.font  = font
        self.text  = init_val
        self.done  = False
        self.cancelled = False

    def handle(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.done = True
            elif event.key == pygame.K_ESCAPE:
                self.cancelled = True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode in "0123456789.-":
                self.text += event.unicode

    def draw(self, surf: pygame.Surface) -> None:
        draw_rrect(surf, INP_BG, self.rect)
        draw_rrect(surf, INP_BD, self.rect, w=2)
        txt = self.font.render(self.text + "|", True, WHITE)
        surf.blit(txt, (self.rect.x + 8, self.rect.y + 6))


# ═══════════════════════════════════════════════════════════════════════════════
#  MENU SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class MenuScreen:
    def __init__(self, surf: pygame.Surface) -> None:
        self.surf = surf
        self.f24  = pygame.font.SysFont("arial", 24)
        self.f18  = pygame.font.SysFont("arial", 18)
        self.f32  = pygame.font.SysFont("arial", 32, bold=True)
        self.f14  = pygame.font.SysFont("arial", 14)

        # Input fields
        cx = SW // 2
        self.fields = {
            "name":     {"label": "Ten nguoi choi:",   "val": "Explorer", "rect": pygame.Rect(cx - 160, 280, 320, 38), "type": "str", "min": None, "max": None},
            "base_hr":  {"label": "Nhip tim co ban (BPM):", "val": "70", "rect": pygame.Rect(cx - 160, 360, 320, 38), "type": "float", "min": 45, "max": 120},
            "base_hrv": {"label": "HRV co ban (ms):",  "val": "62", "rect": pygame.Rect(cx - 160, 440, 320, 38), "type": "float", "min": 20, "max": 100},
        }
        self.active_field: str = ""
        self.start_btn = Button((cx - 100, 530, 200, 48), "BAT DAU CHOI", self.f24)
        self.err = ""

    def handle(self, events: list) -> tuple:
        for e in events:
            # Click on fields
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                self.active_field = ""
                for k, fd in self.fields.items():
                    if fd["rect"].collidepoint(e.pos):
                        self.active_field = k
                        break
            # Keydown
            if e.type == pygame.KEYDOWN:
                if self.active_field:
                    fd = self.fields[self.active_field]
                    if e.key == pygame.K_BACKSPACE:
                        fd["val"] = fd["val"][:-1]
                    elif e.key == pygame.K_TAB:
                        keys = list(self.fields.keys())
                        idx  = keys.index(self.active_field)
                        self.active_field = keys[(idx + 1) % len(keys)]
                    elif e.key == pygame.K_RETURN:
                        self.active_field = ""
                    else:
                        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-_"
                        if e.unicode in allowed:
                            fd["val"] += e.unicode
            # Start button
            if self.start_btn.handle(e):
                return self._try_start()
            elif e.type == pygame.MOUSEMOTION:
                self.start_btn.handle(e)
        return None

    def _try_start(self) -> tuple:
        name = self.fields["name"]["val"].strip() or "Explorer"
        try:
            hr = float(self.fields["base_hr"]["val"])
            if not (45 <= hr <= 120):
                self.err = "Nhip tim co ban: 45–120 BPM"
                return None
        except ValueError:
            self.err = "Nhip tim co ban phai la so!"
            return None
        try:
            hrv = float(self.fields["base_hrv"]["val"])
            if not (20 <= hrv <= 100):
                self.err = "HRV co ban: 20–100 ms"
                return None
        except ValueError:
            self.err = "HRV co ban phai la so!"
            return None
        self.err = ""
        return (name, hr, hrv)

    def draw(self) -> None:
        self.surf.fill(BG)
        # Title
        cx = SW // 2
        t1 = self.f32.render("MO PHONG DU DOAN NGUY HIEM", True, (80, 160, 240))
        t2 = self.f18.render("Su dung mo hinh Random Forest de du doan nguy co sinh ton", True, DIM)
        self.surf.blit(t1, t1.get_rect(centerx=cx, y=140))
        self.surf.blit(t2, t2.get_rect(centerx=cx, y=185))

        # Fields
        for k, fd in self.fields.items():
            lbl = self.f18.render(fd["label"], True, TXT)
            self.surf.blit(lbl, (fd["rect"].x, fd["rect"].y - 24))
            # Hint text below
            if fd["min"] is not None:
                hint = self.f14.render(f"[{fd['min']} - {fd['max']}]", True, DIM)
                self.surf.blit(hint, (fd["rect"].right + 6, fd["rect"].y + 10))
            bg  = INP_BG if k == self.active_field else (20, 28, 50)
            bdr = INP_BD if k == self.active_field else (60, 80, 130)
            draw_rrect(self.surf, bg,  fd["rect"])
            draw_rrect(self.surf, bdr, fd["rect"], w=2)
            cursor = "|" if k == self.active_field else ""
            val_s = self.f24.render(fd["val"] + cursor, True, WHITE)
            self.surf.blit(val_s, (fd["rect"].x + 10, fd["rect"].y + 7))

        # Error
        if self.err:
            es = self.f18.render(self.err, True, RED)
            self.surf.blit(es, es.get_rect(centerx=cx, y=505))

        self.start_btn.draw(self.surf)

        footer = self.f14.render("Nhan Tab de chuyen o nhap | Phim Enter de xac nhan", True, DIM)
        self.surf.blit(footer, footer.get_rect(centerx=cx, y=598))


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class GameScreen:
    CHAR_CX = GAME_W // 2
    CHAR_CY = TOP_H + MAIN_H // 2 + 30

    def __init__(self, surf: pygame.Surface, name: str,
                 base_hr: float, base_hrv: float) -> None:
        self.surf   = surf
        self.name   = name
        self.sim    = PhysioSim(base_hr, base_hrv)
        self.pred   = MLPredictor()
        self.char   = Character(self.CHAR_CX, self.CHAR_CY)

        self.paused = False
        self.dead   = False
        self.dead_timer = 0

        # Font sizes
        self.f22  = pygame.font.SysFont("arial", 22, bold=True)
        self.f18  = pygame.font.SysFont("arial", 18)
        self.f16  = pygame.font.SysFont("arial", 16)
        self.f14  = pygame.font.SysFont("arial", 14)
        self.f13  = pygame.font.SysFont("arial", 13)
        self.f28  = pygame.font.SysFont("arial", 28, bold=True)
        self.f36  = pygame.font.SysFont("arial", 36, bold=True)

        # Preset buttons (top bar)
        pw = GAME_W // len(PRESETS)
        self.preset_btns: list[Button] = []
        for i, key in enumerate(PRESET_KEYS):
            b = Button((i * pw, 0, pw - 2, TOP_H - 4), PRESET_LABELS[key], self.f14)
            self.preset_btns.append(b)
        self.preset_btns[0].active = True

        # Pause button
        self.pause_btn = Button((GAME_W + 5, 4, 90, TOP_H - 8), "PAUSE", self.f16)

        # Action buttons (bottom bar) — 7 buttons
        bw = SW // 7
        bh = BOT_H - 20
        by = SH - BOT_H + 10
        self.act_btns: list[Button] = [
            Button((0*bw+4,  by, bw-8, bh), "Quan ao am +",  self.f16, toggle=False),
            Button((1*bw+4,  by, bw-8, bh), "Quan ao -",     self.f16, toggle=False),
            Button((2*bw+4,  by, bw-8, bh), "Mat na O2",     self.f16, toggle=True),
            Button((3*bw+4,  by, bw-8, bh), "Ngu",           self.f16, toggle=True),
            Button((4*bw+4,  by, bw-8, bh), "Van dong manh", self.f16, toggle=True),
            Button((5*bw+4,  by, bw-8, bh), "Dung day",      self.f16, toggle=False),
            Button((6*bw+4,  by, bw-8, bh), "Uong nuoc",     self.f16, toggle=False),
        ]
        # Sync toggles
        self.act_btns[2].active = self.sim.mask
        self.act_btns[3].active = self.sim.sleeping
        self.act_btns[4].active = self.sim.exercising

        # Environment stat click zones (for editing when paused)
        # These are filled during draw; maps param_key → Rect
        self.env_click_rects: dict = {}
        self.editing_param: str   = ""
        self.text_input: TextInput | None = None

        # Mouse knock tracking
        self.prev_mouse: tuple = pygame.mouse.get_pos()
        self.char_rect  = pygame.Rect(self.CHAR_CX - 40, self.CHAR_CY - 90, 80, 100)

        # Background scroll (parallax)
        self.scroll_x = 0.0

        # Particle systems
        self.particles: list = []   # (x, y, vx, vy, life, max_life, col)
        self.part_timer = 0

        # Suggestion cache
        self.suggestion = "Bat dau hanh trinh..."

    # ── main update ─────────────────────────────────────────────

    def update(self, events: list) -> str:
        # Text input overlay (editing weather param)
        if self.text_input is not None:
            for e in events:
                self.text_input.handle(e)
            if self.text_input.done:
                self._apply_edit()
            if self.text_input.done or self.text_input.cancelled:
                self.text_input = None
                self.editing_param = ""
            return ""

        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_SPACE, pygame.K_p):
                    self.paused = not self.paused
                elif e.key == pygame.K_ESCAPE:
                    return "menu"
                elif e.key == pygame.K_RETURN and self.dead:
                    return "menu"

            if self.pause_btn.handle(e):
                self.paused = not self.paused

            # Preset buttons
            for i, btn in enumerate(self.preset_btns):
                if btn.handle(e):
                    for b in self.preset_btns:
                        b.active = False
                    btn.active = True
                    self.sim.apply_preset(PRESET_KEYS[i])

            # Action buttons
            for idx, btn in enumerate(self.act_btns):
                if btn.handle(e):
                    self._handle_action(idx)

            # Click env param (edit) when paused
            if (e.type == pygame.MOUSEBUTTONDOWN and e.button == 1
                    and self.paused and not self.dead):
                for pkey, rect in self.env_click_rects.items():
                    if rect.collidepoint(e.pos):
                        self.editing_param = pkey
                        cur = getattr(self.sim, "t_" + pkey, getattr(self.sim, pkey, "0"))
                        ir  = pygame.Rect(rect.x, rect.y - 38, 140, 34)
                        self.text_input = TextInput(ir, self.f16, str(round(cur, 1)))
                        break

            # Mouse knock
            if e.type == pygame.MOUSEMOTION and not self.paused and not self.dead:
                cur = e.pos
                dx  = cur[0] - self.prev_mouse[0]
                dy  = cur[1] - self.prev_mouse[1]
                vel = math.sqrt(dx*dx + dy*dy)
                if vel > 6 and self.char_rect.collidepoint(cur):
                    intensity = min(1.0, (vel - 6) / 30.0)
                    self.sim.do_knock(intensity)
                self.prev_mouse = cur

        if self.paused or self.dead:
            if self.dead:
                self.dead_timer += 1
            return ""

        # Physics update
        self.sim.update()

        # ML prediction
        if len(self.sim.buffer) >= WIN_SZ:
            lbl, conf = self.pred.predict(
                self.sim.buffer,
                base_hr=self.sim.base_hr,
                base_hrv=self.sim.base_hrv,
            )
            self.sim.label      = lbl
            self.sim.confidence = conf
            self.suggestion     = get_suggestion(
                lbl, self.sim.hr, self.sim.hrv, self.sim.body_temp,
                self.sim.oxygen, self.sim.aqi, self.sim.trauma,
                self.sim.dehydration,
            )

        # Critical death timer
        if self.sim.label == 2:
            self.sim.crit_frames += 1
        else:
            self.sim.crit_frames = max(0, self.sim.crit_frames - 1)

        if self.sim.crit_frames >= CRIT_DEATH and not self.dead:
            self.dead = True

        # Character animation
        self.char.update(self.sim, self.dead)

        # Background scroll
        spd = max(0.0, self.sim.get_exertion() if hasattr(self.sim, 'get_exertion')
                  else 0.2) + 0.1
        if not self.sim.sleeping and not self.sim.fallen:
            self.scroll_x = (self.scroll_x + spd * 1.2) % GAME_W

        # Particles
        self._update_particles()
        return ""

    def _handle_action(self, idx: int) -> None:
        if   idx == 0: self.sim.add_clothing()
        elif idx == 1: self.sim.remove_clothing()
        elif idx == 2:
            self.sim.mask = self.act_btns[2].active
        elif idx == 3:
            self.sim.sleeping  = self.act_btns[3].active
            if self.sim.sleeping:
                self.act_btns[4].active = False
                self.sim.exercising = False
        elif idx == 4:
            self.sim.exercising = self.act_btns[4].active
            if self.sim.exercising:
                self.act_btns[3].active = False
                self.sim.sleeping = False
        elif idx == 5:
            self.sim.stand_up()
            self.act_btns[3].active = False
            self.sim.sleeping = False
        elif idx == 6:
            self.sim.drink()

    def _apply_edit(self) -> None:
        if not self.editing_param or self.text_input is None:
            return
        try:
            v = float(self.text_input.text)
            mn, mx = PARAM_RANGE[self.editing_param]
            v = max(mn, min(mx, v))
            setattr(self.sim, "t_" + self.editing_param, v)
        except (ValueError, KeyError):
            pass

    # ── particles ───────────────────────────────────────────────

    def _update_particles(self) -> None:
        key = self.sim.terrain
        self.part_timer += 1

        # Spawn
        if key in ("Nui tuyet", "Bac cuc") and self.part_timer % 3 == 0:
            for _ in range(2):
                self.particles.append([
                    random.randint(0, GAME_W), TOP_H,
                    random.uniform(-0.5, 0.5), random.uniform(2, 4),
                    0, 80, (220, 230, 245),
                ])
        elif key == "Chay rung" and self.part_timer % 2 == 0:
            for _ in range(3):
                self.particles.append([
                    random.randint(0, GAME_W),
                    TOP_H + MAIN_H - 80,
                    random.uniform(-1.5, 1.5),
                    random.uniform(-4, -1.5),
                    0, 50, random.choice([(220,80,20),(230,140,20),(200,60,10)]),
                ])

        # Update & expire
        alive = []
        for p in self.particles:
            p[0] += p[2]; p[1] += p[3]; p[4] += 1
            if p[4] < p[5] and TOP_H <= p[1] <= TOP_H + MAIN_H:
                alive.append(p)
        self.particles = alive[:200]

    # ── draw ─────────────────────────────────────────────────────

    def draw(self) -> None:
        self.surf.fill(BG)
        self._draw_terrain()
        self._draw_particles()
        self.char.draw(self.surf, self.sim, self.dead, self.sim.label)
        self._draw_topbar()
        self._draw_stats()
        self._draw_botbar()
        if self.paused and not self.dead:
            self._draw_pause_overlay()
        if self.dead:
            self._draw_death_overlay()
        if self.text_input:
            self.text_input.draw(self.surf)

    def _draw_terrain(self) -> None:
        key = self.sim.terrain
        tc  = TERRAIN_COL.get(key, TERRAIN_COL["Ly tuong"])
        sky_t, sky_b, gnd = tc["sky_t"], tc["sky_b"], tc["gnd"]

        # Sky gradient (approximate with 2 rects)
        sky_rect = pygame.Rect(0, TOP_H, GAME_W, MAIN_H - 80)
        mid = (TOP_H + TOP_H + MAIN_H - 80) // 2
        pygame.draw.rect(self.surf, sky_t, (0, TOP_H, GAME_W, MAIN_H // 2))
        pygame.draw.rect(self.surf, sky_b, (0, TOP_H + MAIN_H // 2, GAME_W, MAIN_H // 2 - 80))

        # Ground
        gy = TOP_H + MAIN_H - 80
        pygame.draw.rect(self.surf, gnd, (0, gy, GAME_W, 80 + BOT_H))

        # Scrolling ground stripes
        stripe_w = 60
        y_stripe = gy + 12
        for i in range(-1, GAME_W // stripe_w + 2):
            sx = int(i * stripe_w - self.scroll_x % stripe_w)
            stripe_col = lerp_col(gnd, (0, 0, 0), 0.08)
            pygame.draw.rect(self.surf, stripe_col,
                             (sx, y_stripe, stripe_w // 2, 12), border_radius=3)

        # Distant hills (silhouette)
        hill_col = lerp_col(sky_b, gnd, 0.35)
        for hx in range(-1, 3):
            bx = int(hx * 300 - (self.scroll_x * 0.15) % 300)
            pts = [(bx, gy), (bx+50, gy-90), (bx+120, gy-50),
                   (bx+180, gy-110), (bx+260, gy-70), (bx+300, gy)]
            pygame.draw.polygon(self.surf, hill_col, pts)

    def _draw_particles(self) -> None:
        for p in self.particles:
            pygame.draw.circle(self.surf, p[6], (int(p[0]), int(p[1])), 3)

    def _draw_topbar(self) -> None:
        pygame.draw.rect(self.surf, TOPBG, (0, 0, SW, TOP_H))
        pygame.draw.line(self.surf, (50, 65, 110), (0, TOP_H - 1), (SW, TOP_H - 1), 2)

        for btn in self.preset_btns:
            btn.draw(self.surf)

        # Player name + label badge (right of game area, left of stats)
        nm = self.f16.render(f"  {self.name}  ", True, TXT)
        self.surf.blit(nm, nm.get_rect(x=GAME_W + 105, centery=TOP_H // 2))

        # Pause button
        self.pause_btn.label = "RESUME" if self.paused else "PAUSE"
        self.pause_btn.draw(self.surf)

    def _draw_stats(self) -> None:
        s   = self.sim
        sx  = GAME_W
        sy  = TOP_H
        sw  = STAT_W
        pad = 6

        # ── Physiological panel ──────────────────────────────────
        ph = 176
        draw_panel(self.surf, (sx+pad, sy+pad, sw-pad*2, ph),
                   PHY_BG, PHY_BD, "CHI SO SINH LY", self.f16, bdr_w=2)
        self._stat_row(sx+pad+8, sy+34, "Nhip tim (HR)",  f"{s.hr:.1f} BPM",
                       s.hr, [(55,195),(120,155),(160,115),(200, 65)], [60, 100, 140, 175])
        self._stat_row(sx+pad+8, sy+70, "HRV",            f"{s.hrv:.1f} ms",
                       s.hrv, [(195,55),(155,120),(115,155),(65,195)], [15, 30, 50, 80], invert=True)
        self._stat_row(sx+pad+8, sy+106, "Than nhiet",    f"{s.body_temp:.2f} C",
                       s.body_temp, [(55,115,195),(55,195,75),(195,175,25),(195,55,55)],
                       [34.5, 36.0, 37.5, 39.0], three_col=True)
        # Clothing bar
        cl_y = sy + 132
        self.surf.blit(self.f14.render(f"Quan ao: {s.clothing*100:.0f}%", True, DIM),
                       (sx+pad+8, cl_y))
        bar_r = pygame.Rect(sx+pad+120, cl_y+3, 200, 9)
        draw_rrect(self.surf, (40, 50, 80), bar_r)
        draw_rrect(self.surf, (80, 150, 230),
                   (bar_r.x, bar_r.y, int(bar_r.w * s.clothing), bar_r.h))

        # Dehydration bar
        dh_y = sy + 148
        dh_col = (
            RED    if s.dehydration > 0.7 else
            YELLOW if s.dehydration > 0.4 else
            GREEN
        )
        self.surf.blit(self.f14.render(f"Mat nuoc: {s.dehydration*100:.0f}%", True, DIM),
                       (sx+pad+8, dh_y))
        bar_d = pygame.Rect(sx+pad+120, dh_y+3, 200, 9)
        draw_rrect(self.surf, (40, 50, 80), bar_d)
        draw_rrect(self.surf, dh_col,
                   (bar_d.x, bar_d.y, int(bar_d.w * s.dehydration), bar_d.h))

        # ── Environment panel ────────────────────────────────────
        ey  = sy + ph + pad * 2
        eh  = 200
        draw_panel(self.surf, (sx+pad, ey, sw-pad*2, eh),
                   ENV_BG, ENV_BD, "MOI TRUONG", self.f16, bdr_w=2)

        env_items = [
            ("Nhiet do MT",   f"{s.env_temp:.1f} C",   "env_temp"),
            ("Do am",         f"{s.humidity:.1f} %",   "humidity"),
            ("Ap suat",       f"{s.pressure:.0f} hPa", "pressure"),
            ("Nong do Oxy",   f"{s.oxygen:.2f} %",     "oxygen"),
            ("Chi so UV",     f"{s.uv_index:.1f}",     "uv_index"),
            ("Chi so AQI",    f"{s.aqi:.0f}",          "aqi"),
        ]
        self.env_click_rects = {}
        for i, (lbl, val, key) in enumerate(env_items):
            row = i // 2
            col = i % 2
            rx  = sx + pad + 10 + col * ((sw - pad*2) // 2)
            ry  = ey + 32 + row * 52
            # Value color
            vc  = self._env_val_color(key, getattr(s, key))
            lbl_s = self.f13.render(lbl + ":", True, DIM)
            val_s = self.f18.render(val, True, vc)
            self.surf.blit(lbl_s, (rx, ry))
            self.surf.blit(val_s, (rx, ry + 16))
            # Click zone
            cr = pygame.Rect(rx, ry, 190, 42)
            self.env_click_rects[key] = cr
            if self.paused:
                draw_rrect(self.surf, (0, 0, 0, 0), cr, w=1)
                edit_hint = self.f13.render("[click]", True, (60, 80, 130))
                self.surf.blit(edit_hint, (rx + 115, ry + 18))

        # ── Prediction panel ─────────────────────────────────────
        py   = ey + eh + pad * 2
        ph2  = SH - BOT_H - py - pad
        lbl  = self.sim.label
        pbg  = PRED_BG[lbl]
        pbd  = PRED_BD[lbl]
        draw_panel(self.surf, (sx+pad, py, sw-pad*2, ph2),
                   pbg, pbd, "DU DOAN NGUY CO", self.f16, bdr_w=2)

        # Big label
        lname = LABEL_NAME[lbl]
        lcol  = LABEL_COL[lbl]
        ls    = self.f28.render(lname, True, lcol)
        self.surf.blit(ls, ls.get_rect(centerx=sx + sw//2, y=py + 30))

        # Confidence bar
        if self.sim.confidence > 0:
            conf_y = py + 62
            bar_bg = pygame.Rect(sx+pad+8, conf_y, sw-pad*2-16, 10)
            draw_rrect(self.surf, (30, 40, 65), bar_bg)
            bar_fill = pygame.Rect(bar_bg.x, bar_bg.y,
                                   int(bar_bg.w * self.sim.confidence), bar_bg.h)
            draw_rrect(self.surf, lcol, bar_fill)
            cf_txt = self.f13.render(f"Do tin cay: {self.sim.confidence*100:.0f}%", True, DIM)
            self.surf.blit(cf_txt, (bar_bg.x, conf_y + 12))

        # Suggestion (wrapped)
        lines = wrap_lines(self.suggestion, self.f14, sw - pad*2 - 20)
        for j, ln in enumerate(lines[:5]):
            lc = lcol if lbl > 0 else TXT
            ls = self.f14.render(ln, True, lc)
            self.surf.blit(ls, (sx+pad+10, py + 82 + j * 18))

        # Critical timer bar
        if lbl == 2:
            ct_y = py + ph2 - 28
            bar_r2 = pygame.Rect(sx+pad+8, ct_y, sw-pad*2-16, 14)
            draw_rrect(self.surf, (60, 15, 15), bar_r2)
            frac = min(1.0, self.sim.crit_frames / CRIT_DEATH)
            draw_rrect(self.surf, RED,
                       (bar_r2.x, bar_r2.y, int(bar_r2.w * frac), bar_r2.h))
            self.surf.blit(self.f13.render("Tu vong:", True, RED), (bar_r2.x, ct_y - 16))

    def _stat_row(self, x: int, y: int, label: str, val_str: str,
                  val: float, colors, thresholds, invert: bool = False,
                  three_col: bool = False) -> None:
        self.surf.blit(self.f14.render(label + ":", True, DIM), (x, y))
        # Determine color
        if three_col:
            # colors = [cold_col, ok_col, warm_col, hot_col]
            if   val < thresholds[0]: vc = RED
            elif val < thresholds[1]: vc = (55, 115, 195)
            elif val < thresholds[2]: vc = GREEN
            elif val < thresholds[3]: vc = YELLOW
            else:                     vc = RED
        else:
            g, r = colors[0], colors[1]
            if   val < thresholds[0]: vc = GREEN if not invert else RED
            elif val < thresholds[1]: vc = GREEN
            elif val < thresholds[2]: vc = YELLOW
            elif val < thresholds[3]: vc = RED
            else:                     vc = RED
        vs = self.f18.render(val_str, True, vc)
        self.surf.blit(vs, (x + 170, y - 2))

    def _env_val_color(self, key: str, val: float) -> tuple:
        if key == "env_temp":
            if val < -10: return (55, 115, 230)
            if val < 10:  return (130, 175, 230)
            if val < 30:  return GREEN
            if val < 38:  return YELLOW
            return RED
        if key == "oxygen":
            if val < 14:  return RED
            if val < 17:  return YELLOW
            return GREEN
        if key == "aqi":
            if val < 50:  return GREEN
            if val < 150: return YELLOW
            return RED
        return TXT

    def _draw_botbar(self) -> None:
        pygame.draw.rect(self.surf, BOTBG, (0, SH - BOT_H, SW, BOT_H))
        pygame.draw.line(self.surf, (50, 65, 110), (0, SH - BOT_H), (SW, SH - BOT_H), 2)

        # Sync toggle states
        self.act_btns[2].active = self.sim.mask
        self.act_btns[3].active = self.sim.sleeping
        self.act_btns[4].active = self.sim.exercising
        # Tint "Uong nuoc" button theo mức dehydration
        self.act_btns[6].label = (
            "Uong nuoc (!)" if self.sim.dehydration > 0.5 else "Uong nuoc"
        )

        for btn in self.act_btns:
            btn.draw(self.surf)

        # Clothing visual indicator on btn0/1
        cl_s = self.f13.render(f"{self.sim.clothing*100:.0f}%", True, DIM)
        self.surf.blit(cl_s, (8, SH - 22))

    def _draw_pause_overlay(self) -> None:
        ov = pygame.Surface((GAME_W, MAIN_H), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 100))
        self.surf.blit(ov, (0, TOP_H))
        ps = self.f36.render("TAM DUNG", True, WHITE)
        self.surf.blit(ps, ps.get_rect(centerx=GAME_W//2, centery=TOP_H + MAIN_H//2))
        hs = self.f16.render("Click thong so moi truong de chinh sua | SPACE de tiep tuc", True, DIM)
        self.surf.blit(hs, hs.get_rect(centerx=GAME_W//2, centery=TOP_H + MAIN_H//2 + 46))

    def _draw_death_overlay(self) -> None:
        ov = pygame.Surface((SW, SH), pygame.SRCALPHA)
        alpha = min(200, self.dead_timer * 4)
        ov.fill((120, 0, 0, alpha))
        self.surf.blit(ov, (0, 0))
        if self.dead_timer > 20:
            ds = self.f36.render("GAME OVER", True, RED)
            ns = self.f22.render(f"{self.name} da khong the tiep tuc hanh trinh.", True, WHITE)
            rs = self.f18.render("Nhan Enter hoac ESC de ve menu", True, DIM)
            cx = SW // 2
            self.surf.blit(ds, ds.get_rect(centerx=cx, centery=SH//2 - 50))
            self.surf.blit(ns, ns.get_rect(centerx=cx, centery=SH//2))
            self.surf.blit(rs, rs.get_rect(centerx=cx, centery=SH//2 + 46))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Mo Phong Du Doan Nguy Hiem v6")
    clock = pygame.time.Clock()

    state = "menu"
    menu:  MenuScreen | None = MenuScreen(screen)
    game:  GameScreen | None = None

    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if state == "menu":
            result = menu.handle(events)
            menu.draw()
            if result:
                name, hr, hrv = result
                game  = GameScreen(screen, name, hr, hrv)
                state = "game"

        elif state == "game":
            assert game is not None
            result = game.update(events)
            game.draw()
            if result == "menu":
                menu  = MenuScreen(screen)
                game  = None
                state = "menu"

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()