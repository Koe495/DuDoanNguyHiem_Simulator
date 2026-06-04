# genData7.py
"""
Dataset Generator v6v2 – Survival Risk Simulation (Updated)
============================================================
Thay đổi so với phiên bản trước:
  - TIME_STEP_S = 0.2s  (khớp temporal resolution với game6.py / SAMPLE_EVERY=6 @ 30fps)
  - Phương trình sinh lý được co giãn theo SCALE = 0.2/5 = 0.04
  - Nhãn có vùng mờ (fuzzy label) quanh ngưỡng phân loại
  - Dehydration tích hợp thêm vào kịch bản CRIT_HEATSTROKE, CRIT_DESERT_HEAT
  - 6 kịch bản cross-scenario (chuyển tiếp giữa 2 trạng thái)
  - Xuất: simulation_survival_dataset6v2.csv + scenarios.json
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

RANDOM_SEED   = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Hằng số thời gian ────────────────────────────────────────────────────────
TIME_STEP_S = 0.2               # Bước thời gian (giây) – khớp game6.py
SCALE       = TIME_STEP_S / 5.0 # 0.04 – hệ số co giãn phương trình sinh lý

# ── Hằng số phân loại ────────────────────────────────────────────────────────
CRIT_THRESH  = 4.0
WARN_THRESH  = 1.2
NOISE_ZONE   = 0.30             # Vùng mờ ±0.30 quanh mỗi ngưỡng

# ── File đầu ra ───────────────────────────────────────────────────────────────
OUTPUT_CSV  = "simulation_survival_dataset7.csv"
OUTPUT_JSON = "scenarios.json"


# ═══════════════════════════════════════════════════════════════════════════════
#  Tham số môi trường ban đầu cho mỗi kịch bản
# ═══════════════════════════════════════════════════════════════════════════════

def _init_env(name: str) -> dict:
    """
    Trả về dict tham số môi trường / sinh lý khởi đầu cho kịch bản `name`.
    dehy_rate: mức tăng dehydration mỗi bước TIME_STEP_S (đã co giãn từ /5s).
    trauma_rate: mức tăng trauma_shock mỗi bước (đã co giãn).
    """
    p = dict(
        exertion    = 0.1,
        clothing    = 0.5,
        oxygen_env  = 20.9,
        aqi_env     = 25.0,
        env_temp    = 24.0,
        humidity    = random.uniform(50.0, 65.0),
        pressure    = random.uniform(1005.0, 1013.0),
        uv_index    = random.uniform(1.0, 3.0),
        dehy_rate   = 0.0,
        trauma_rate = 0.0,
    )

    if name == "NORMAL_WALKING":
        p["exertion"]  = random.uniform(0.15, 0.3)
        p["env_temp"]  = random.uniform(18.0, 26.0)

    elif name == "NORMAL_HEAVY_CLIMB":
        p["exertion"]  = random.uniform(0.75, 0.9)
        p["env_temp"]  = random.uniform(22.0, 28.0)

    elif name == "NORMAL_JUMPING":
        p["exertion"]  = 0.4
        p["env_temp"]  = random.uniform(20.0, 25.0)

    elif name == "WARN_INTERNAL_INJURY":
        p["exertion"]    = 0.05
        p["env_temp"]    = random.uniform(15.0, 22.0)
        p["trauma_rate"] = 0.008 * SCALE    # 0.00032 /step

    elif name == "WARN_SILENT_POISON":
        p["aqi_env"]    = random.uniform(180.0, 280.0)
        p["oxygen_env"] = random.uniform(17.0, 18.5)

    elif name == "WARN_SLEEP_HYPOTHERMIA":
        p["exertion"]  = -0.2
        p["env_temp"]  = random.uniform(-10.0, 2.0)
        p["clothing"]  = 0.25

    elif name == "WARN_DEHYDRATION":
        p["exertion"]  = random.uniform(0.3, 0.5)
        p["env_temp"]  = random.uniform(30.0, 36.0)
        p["humidity"]  = random.uniform(20.0, 35.0)
        p["dehy_rate"] = 0.002  # ~0.5 dehydration sau 250 bước (50s)

    elif name == "CRIT_SEVERE_FALL":
        p["exertion"]    = -0.3
        p["env_temp"]    = random.uniform(10.0, 18.0)
        p["trauma_rate"] = 0.02 * SCALE   # 0.0008 /step

    elif name == "CRIT_HEATSTROKE":
        p["exertion"]  = 0.7
        p["env_temp"]  = random.uniform(39.0, 45.0)
        p["clothing"]  = 0.4
        p["uv_index"]  = random.uniform(8.0, 11.0)
        p["dehy_rate"] = 0.0015  # mất nước nhanh vì gắng sức + nóng

    elif name == "CRIT_DESERT_HEAT":
        p["exertion"]  = random.uniform(0.2, 0.5)
        p["env_temp"]  = random.uniform(38.0, 44.0)
        p["clothing"]  = random.uniform(0.1, 0.3)
        p["uv_index"]  = random.uniform(8.0, 11.5)
        p["humidity"]  = random.uniform(5.0, 18.0)
        p["aqi_env"]   = random.uniform(20.0, 55.0)
        p["dehy_rate"] = 0.001  # sa mạc khô nóng

    elif name == "CRIT_TOXIC_COLLAPSE":
        p["exertion"]    = -0.2
        p["aqi_env"]     = random.uniform(350.0, 500.0)
        p["oxygen_env"]  = random.uniform(14.0, 16.5)
        p["trauma_rate"] = 0.015 * SCALE   # nhiễm độc gây shock

    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  Bộ phân loại lâm sàng + fuzzy label
# ═══════════════════════════════════════════════════════════════════════════════

def _derive_label(body_temp: float, oxygen_env: float, aqi_env: float,
                  hrv: float, fall_duration: float, trauma_shock: float,
                  dehydration: float, uv_index: float, env_temp: float,
                  base_hr: float = 70.0, base_hrv: float = 62.0,
                  hr: float = 70.0) -> int:
    """
    Tính biological_deviance cá nhân hoá rồi gán nhãn có vùng mờ.

    Yếu tố môi trường (oxy, AQI, UV, nhiệt, té ngã) → ngưỡng tuyệt đối.
    HR và HRV → ngưỡng tương đối so với baseline cá nhân.
    """
    dev = 0.0

    # ── Môi trường / tuyệt đối ───────────────────────────────────────────────
    dev += (abs(body_temp - 36.8) / 1.3) ** 2
    dev += (max(0.0, 20.9 - oxygen_env) / 2.0) ** 2
    dev += (max(0.0, aqi_env - 120.0) / 100.0) ** 2

    if fall_duration > 15:
        dev += fall_duration / 25.0
    if trauma_shock > 0.2:
        dev += trauma_shock * 3.5
    if dehydration > 0.5:
        dev += dehydration * 1.5

    uv_heat = (max(0.0, uv_index - 5.0) / 7.0
               + max(0.0, env_temp - 32.0) / 12.0)
    dev += uv_heat * (1.0 + max(0.0, body_temp - 37.0) / 2.0)

    # ── HRV cá nhân hoá: cảnh báo khi HRV < 65% baseline cá nhân ────────────
    hrv_thresh = base_hrv * 0.65
    hrv_norm   = max(base_hrv * 0.25, 4.0)
    dev += (max(0.0, hrv_thresh - hrv) / hrv_norm) ** 2

    # ── HR cá nhân hoá: cảnh báo khi HR tăng > 55 BPM so với baseline ────────
    hr_high_thresh = base_hr + 55.0
    dev += (max(0.0, hr - hr_high_thresh) / 25.0) ** 2

    # ── Fuzzy labeling ───────────────────────────────────────────────────────
    hi, lo = CRIT_THRESH, WARN_THRESH

    if dev >= hi + NOISE_ZONE:
        return 2
    if dev >= hi - NOISE_ZONE:
        t = (dev - (hi - NOISE_ZONE)) / (2 * NOISE_ZONE)
        return 2 if random.random() < t else 1

    if dev >= lo + NOISE_ZONE:
        return 1
    if dev >= lo - NOISE_ZONE:
        t = (dev - (lo - NOISE_ZONE)) / (2 * NOISE_ZONE)
        return 1 if random.random() < t else 0

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Bước sinh lý một TIME_STEP_S (đã co giãn)
# ═══════════════════════════════════════════════════════════════════════════════

def _physio_step(hr: float, hrv: float, body_temp: float,
                 dehydration: float, trauma_shock: float,
                 base_hr: float, base_hrv: float,
                 exertion: float, env_temp: float, clothing: float,
                 oxygen_env: float, aqi_env: float) -> tuple:
    """
    Tiến một bước TIME_STEP_S. Trả về (hr, hrv, body_temp).
    Không cập nhật dehydration / trauma_shock (do caller quản lý).
    """
    # A. Thân nhiệt
    hp = 0.012 * (1.0 + exertion) * SCALE
    hl = 0.004 * (env_temp - body_temp) * (1.0 - clothing) * SCALE
    dh = dehydration * 0.005 * SCALE
    body_temp = float(np.clip(
        body_temp + hp + hl + dh + random.gauss(0, 0.004),
        30.0, 43.0
    ))

    # B. Nhịp tim
    hypoxia_d     = max(0.0, 20.9 - oxygen_env) * 4.5
    toxic_d       = max(0.0, aqi_env - 100.0) * 0.04
    dehy_d        = dehydration * 20.0
    shock_d       = trauma_shock * 40.0 if exertion >= 0 else -trauma_shock * 25.0
    target_hr     = base_hr + exertion * 65.0 + hypoxia_d + toxic_d + shock_d + dehy_d
    hr = float(np.clip(
        hr + (target_hr - hr) * 0.12 * SCALE + random.gauss(0, 0.24),
        35.0, 195.0
    ))

    # C. HRV
    thermal_s = abs(body_temp - 36.8) * 16.0
    hypoxia_s = max(0.0, 20.9 - oxygen_env) * 10.0
    toxic_s   = max(0.0, aqi_env - 50.0) * 0.12
    injury_s  = trauma_shock * 45.0
    dehy_s    = dehydration * 15.0
    target_hrv = (base_hrv - exertion * 20.0
                  - thermal_s - hypoxia_s - toxic_s - injury_s - dehy_s)
    hrv = float(np.clip(
        hrv + (target_hrv - hrv) * 0.08 * SCALE + random.gauss(0, 0.16),
        4.0, 100.0
    ))

    return hr, hrv, body_temp


# ═══════════════════════════════════════════════════════════════════════════════
#  Mô phỏng kịch bản đơn
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_user_timeline(scenario_name: str, num_events: int,
                           rows_per_event: int) -> list:
    """Mô phỏng chuỗi thời gian với bước TIME_STEP_S cho kịch bản đơn."""
    data = []

    for ev in range(num_events):
        user_id   = f"User_{scenario_name}_{ev + 1:03d}"
        base_time = datetime.now() - timedelta(days=ev)

        base_hr  = random.uniform(50.0, 90.0)   # cá nhân hoá: 50–90 BPM
        base_hrv = random.uniform(25.0, 90.0)   # cá nhân hoá: 25–90 ms
        hr, hrv, body_temp = base_hr, base_hrv, 36.8

        p             = _init_env(scenario_name)
        exertion      = p["exertion"]
        clothing      = p["clothing"]
        oxygen_env    = p["oxygen_env"]
        aqi_env       = p["aqi_env"]
        env_temp      = p["env_temp"]
        humidity      = p["humidity"]
        pressure      = p["pressure"]
        uv_index      = p["uv_index"]
        dehy_rate     = p["dehy_rate"]
        trauma_rate   = p["trauma_rate"]

        dehydration   = 0.0
        trauma_shock  = 0.0
        fall_duration = 0

        # Số bước tương đương "frame gốc" để xác định sự kiện nhảy/ngã
        # (sự kiện cũ ở bước 5s, nay mỗi "frame cũ" = 25 bước 0.2s)
        FRAME_SCALE = 25

        for i in range(rows_per_event):
            current_time = base_time + timedelta(seconds=i * TIME_STEP_S)
            accel = random.uniform(0.95, 1.05) + max(0.0, exertion) * 0.1

            old_frame = i // FRAME_SCALE   # bước tương đương gốc

            # ── Sự kiện đặc thù kịch bản ─────────────────────────────────────
            if scenario_name == "NORMAL_JUMPING" and (i % FRAME_SCALE == 0) and old_frame in [20, 50, 80]:
                accel = random.uniform(3.6, 5.2)

            elif scenario_name == "WARN_INTERNAL_INJURY":
                if i == 0:
                    accel = random.uniform(4.5, 6.0)
                trauma_shock = min(0.6, trauma_shock + trauma_rate)

            elif scenario_name == "WARN_DEHYDRATION":
                dehydration = min(1.0, dehydration + dehy_rate)

            elif scenario_name == "CRIT_SEVERE_FALL":
                if i == 0:
                    accel = random.uniform(6.5, 9.0)
                fall_duration = int(i * TIME_STEP_S)
                trauma_shock  = min(1.0, trauma_shock + trauma_rate)

            elif scenario_name == "CRIT_HEATSTROKE":
                dehydration = min(1.0, dehydration + dehy_rate)

            elif scenario_name == "CRIT_DESERT_HEAT":
                dehydration = min(1.0, dehydration + dehy_rate)

            elif scenario_name == "CRIT_TOXIC_COLLAPSE":
                if i == 0:
                    accel = random.uniform(2.0, 3.5)
                trauma_shock = min(0.8, trauma_shock + trauma_rate)
                oxygen_env   = max(12.0, oxygen_env - 0.05 * SCALE)

            # ── Cập nhật sinh lý ──────────────────────────────────────────────
            hr, hrv, body_temp = _physio_step(
                hr, hrv, body_temp, dehydration, trauma_shock,
                base_hr, base_hrv, exertion, env_temp, clothing,
                oxygen_env, aqi_env,
            )

            derived_label = _derive_label(
                body_temp, oxygen_env, aqi_env, hrv,
                fall_duration, trauma_shock, dehydration, uv_index, env_temp,
                base_hr=base_hr, base_hrv=base_hrv, hr=hr,
            )

            data.append([
                current_time, user_id,
                round(hr, 1), round(hrv, 1), round(body_temp, 2),
                round(env_temp, 1), round(humidity, 1), round(pressure, 1),
                round(oxygen_env, 2), round(uv_index, 1), int(aqi_env),
                round(accel, 2), fall_duration,
                round(base_hr, 1), round(base_hrv, 1),   # baseline cá nhân
                derived_label,
            ])

    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  Mô phỏng kịch bản cross-scenario (chuyển tiếp)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_composite_scenario(scenario_a: str, scenario_b: str,
                                num_events: int, rows_per_event: int,
                                transition_at: int = None) -> list:
    """
    Nửa đầu: tham số kịch bản A.
    Từ `transition_at`: blend tuyến tính sang kịch bản B trong 1/4 thời gian,
    rồi chạy hoàn toàn B ở 1/4 cuối.

    Trạng thái sinh lý (hr, hrv, body_temp, dehydration, trauma_shock)
    được tiếp tục liên tục qua điểm chuyển đổi.
    """
    if transition_at is None:
        transition_at = rows_per_event // 2

    blend_end = transition_at + rows_per_event // 4

    data = []

    for ev in range(num_events):
        label_a = scenario_a.split("_")[0]  # NORMAL / WARN / CRIT
        label_b = scenario_b.split("_")[0]
        user_id   = f"User_X_{label_a[:3]}_{label_b[:3]}_{ev + 1:03d}"
        base_time = datetime.now() - timedelta(days=ev)

        base_hr  = random.uniform(50.0, 90.0)   # cá nhân hoá: 50–90 BPM
        base_hrv = random.uniform(25.0, 90.0)   # cá nhân hoá: 25–90 ms
        hr, hrv, body_temp = base_hr, base_hrv, 36.8

        pa = _init_env(scenario_a)
        pb = _init_env(scenario_b)

        exertion     = pa["exertion"]
        clothing     = pa["clothing"]
        oxygen_env   = pa["oxygen_env"]
        aqi_env      = pa["aqi_env"]
        env_temp     = pa["env_temp"]
        humidity     = pa["humidity"]
        pressure     = pa["pressure"]
        uv_index     = pa["uv_index"]
        dehy_rate    = pa["dehy_rate"]
        trauma_rate  = pa["trauma_rate"]

        dehydration  = 0.0
        trauma_shock = 0.0
        fall_duration = 0

        for i in range(rows_per_event):
            current_time = base_time + timedelta(seconds=i * TIME_STEP_S)

            # ── Blend tham số kịch bản ────────────────────────────────────────
            if i >= transition_at:
                t = min(1.0, (i - transition_at) / max(1, blend_end - transition_at))

                def lerp(a, b):
                    return a + (b - a) * t

                exertion   = lerp(pa["exertion"],   pb["exertion"])
                clothing   = lerp(pa["clothing"],   pb["clothing"])
                oxygen_env = lerp(pa["oxygen_env"], pb["oxygen_env"])
                aqi_env    = lerp(pa["aqi_env"],    pb["aqi_env"])
                env_temp   = lerp(pa["env_temp"],   pb["env_temp"])
                uv_index   = lerp(pa["uv_index"],   pb["uv_index"])
                dehy_rate  = lerp(pa["dehy_rate"],  pb["dehy_rate"])
                trauma_rate= lerp(pa["trauma_rate"],pb["trauma_rate"])

            # ── Sự kiện đặc biệt từ scenario_b sau điểm chuyển ───────────────
            if i == transition_at:
                # Nếu B là fall: gây ngã đột ngột
                if "FALL" in scenario_b:
                    accel = random.uniform(5.0, 8.0)
                    fall_duration = 0
                elif "TOXIC" in scenario_b:
                    accel = random.uniform(2.0, 3.5)
                else:
                    accel = random.uniform(0.95, 1.05) + max(0.0, exertion) * 0.1
            else:
                accel = random.uniform(0.95, 1.05) + max(0.0, exertion) * 0.1

            # Tích lũy trạng thái liên tục
            dehydration  = min(1.0, dehydration  + dehy_rate)
            trauma_shock = min(1.0, trauma_shock + trauma_rate)
            if i >= transition_at and "FALL" in scenario_b:
                fall_duration = int((i - transition_at) * TIME_STEP_S)
                trauma_shock  = min(1.0, trauma_shock + 0.0008)

            # Oxygen giảm dần nếu B là toxic
            if i >= transition_at and "TOXIC" in scenario_b:
                oxygen_env = max(12.0, oxygen_env - 0.002)

            hr, hrv, body_temp = _physio_step(
                hr, hrv, body_temp, dehydration, trauma_shock,
                base_hr, base_hrv, exertion, env_temp, clothing,
                oxygen_env, aqi_env,
            )

            derived_label = _derive_label(
                body_temp, oxygen_env, aqi_env, hrv,
                fall_duration, trauma_shock, dehydration, uv_index, env_temp,
                base_hr=base_hr, base_hrv=base_hrv, hr=hr,
            )

            data.append([
                current_time, user_id,
                round(hr, 1), round(hrv, 1), round(body_temp, 2),
                round(env_temp, 1), round(humidity, 1), round(pressure, 1),
                round(oxygen_env, 2), round(uv_index, 1), int(aqi_env),
                round(accel, 2), fall_duration,
                round(base_hr, 1), round(base_hrv, 1),   # baseline cá nhân
                derived_label,
            ])

    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  Xuất scenarios.json
# ═══════════════════════════════════════════════════════════════════════════════

def _write_scenarios_json(generation_summary: dict) -> None:
    scenarios_meta = {
        "version"          : "6v2",
        "generated"        : datetime.now().isoformat(timespec="seconds"),
        "time_step_s"      : TIME_STEP_S,
        "recommended_window_size"  : 50,
        "recommended_stride"       : 25,
        "window_coverage_s"        : 50 * TIME_STEP_S,
        "label_thresholds": {
            "safe_to_warn" : WARN_THRESH,
            "warn_to_crit" : CRIT_THRESH,
            "noise_zone"   : NOISE_ZONE,
        },
        "scenarios": {
            "NORMAL_WALKING": {
                "type": "simple", "label_target": 0,
                "description": "Đi bộ bình thường, exertion thấp, nhiệt độ ôn hòa.",
            },
            "NORMAL_HEAVY_CLIMB": {
                "type": "simple", "label_target": 0,
                "description": "Leo núi nặng, exertion cao nhưng môi trường an toàn.",
            },
            "NORMAL_JUMPING": {
                "type": "simple", "label_target": 0,
                "description": "Nhảy định kỳ, accel spike ngắn hạn, ổn định.",
            },
            "WARN_INTERNAL_INJURY": {
                "type": "simple", "label_target": 1,
                "description": "Chấn thương nội tạng tích lũy, trauma_shock tăng dần.",
            },
            "WARN_SILENT_POISON": {
                "type": "simple", "label_target": 1,
                "description": "Môi trường AQI cao và oxy thấp âm thầm.",
            },
            "WARN_SLEEP_HYPOTHERMIA": {
                "type": "simple", "label_target": 1,
                "description": "Ngủ trong môi trường lạnh, quần áo mỏng.",
            },
            "WARN_DEHYDRATION": {
                "type": "simple", "label_target": 1,
                "description": "Gắng sức trong nóng hanh, mất nước tích lũy nhanh.",
            },
            "CRIT_SEVERE_FALL": {
                "type": "simple", "label_target": 2,
                "description": "Ngã mạnh, trauma_shock leo thang nhanh.",
            },
            "CRIT_HEATSTROKE": {
                "type": "simple", "label_target": 2,
                "description": "Say nắng cấp tính: exertion + UV + nhiệt + mất nước.",
            },
            "CRIT_TOXIC_COLLAPSE": {
                "type": "simple", "label_target": 2,
                "description": "Ngộ độc khí: AQI cực cao, oxy giảm dần, shock.",
            },
            "CRIT_DESERT_HEAT": {
                "type": "simple", "label_target": 2,
                "description": "Sa mạc: cực nóng, UV cao, độ ẩm cực thấp, mất nước.",
            },
            "CROSS_WALK_DEHYDRATION": {
                "type": "composite", "label_target": "0→1",
                "scenario_a": "NORMAL_WALKING", "scenario_b": "WARN_DEHYDRATION",
                "description": "Đi bộ trong nóng, dần chuyển sang trạng thái mất nước.",
            },
            "CROSS_CLIMB_INJURY": {
                "type": "composite", "label_target": "0→1",
                "scenario_a": "NORMAL_HEAVY_CLIMB", "scenario_b": "WARN_INTERNAL_INJURY",
                "description": "Leo núi gắng sức rồi bị chấn thương.",
            },
            "CROSS_HYPOTHERMIA_FALL": {
                "type": "composite", "label_target": "1→2",
                "scenario_a": "WARN_SLEEP_HYPOTHERMIA", "scenario_b": "CRIT_SEVERE_FALL",
                "description": "Ngủ lạnh rồi ngã, suy sụp nhanh.",
            },
            "CROSS_DESERT_HEATSTROKE": {
                "type": "composite", "label_target": "2→2",
                "scenario_a": "CRIT_DESERT_HEAT", "scenario_b": "CRIT_HEATSTROKE",
                "description": "Sa mạc leo thang thành say nắng hoàn toàn.",
            },
            "CROSS_WALK_TOXIC": {
                "type": "composite", "label_target": "0→2",
                "scenario_a": "NORMAL_WALKING", "scenario_b": "CRIT_TOXIC_COLLAPSE",
                "description": "Đi bộ rồi tiến vào vùng khí độc, sụp đổ nhanh.",
            },
            "CROSS_DEHYDRATION_HEATSTROKE": {
                "type": "composite", "label_target": "1→2",
                "scenario_a": "WARN_DEHYDRATION", "scenario_b": "CRIT_HEATSTROKE",
                "description": "Mất nước leo thang thành say nắng nguy kịch.",
            },
        },
        "generation_summary": generation_summary,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(scenarios_meta, f, ensure_ascii=False, indent=2)

    print(f"📋 Đã xuất metadata kịch bản → {OUTPUT_JSON}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== BẮT ĐẦU KHỞI TẠO DATASET PHẢN HỒI SINH LÝ V6v2 ===")
    print(f"    Bước thời gian: {TIME_STEP_S}s  |  SCALE={SCALE:.3f}\n")

    all_data: list = []
    summary: dict  = {}

    # ── Nhóm An toàn ──────────────────────────────────────────────────────────
    print("[1/3] Mô phỏng trạng thái an toàn...")
    configs_safe = [
        ("NORMAL_WALKING",    65, 1000),
        ("NORMAL_HEAVY_CLIMB",20, 1000),
        ("NORMAL_JUMPING",    15, 1000),
    ]
    for name, n_ev, n_rows in configs_safe:
        rows = simulate_user_timeline(name, n_ev, n_rows)
        all_data.extend(rows)
        summary[name] = {"num_events": n_ev, "rows_per_event": n_rows, "total_rows": len(rows)}
        print(f"    {name}: {len(rows):,} dòng")

    # ── Nhóm Cảnh báo ─────────────────────────────────────────────────────────
    print("\n[2/3] Mô phỏng trạng thái cảnh báo...")
    configs_warn = [
        ("WARN_INTERNAL_INJURY",   20, 500),
        ("WARN_SILENT_POISON",     20, 500),
        ("WARN_SLEEP_HYPOTHERMIA", 20, 500),
        ("WARN_DEHYDRATION",       20, 500),
    ]
    for name, n_ev, n_rows in configs_warn:
        rows = simulate_user_timeline(name, n_ev, n_rows)
        all_data.extend(rows)
        summary[name] = {"num_events": n_ev, "rows_per_event": n_rows, "total_rows": len(rows)}
        print(f"    {name}: {len(rows):,} dòng")

    # ── Nhóm Nguy kịch ────────────────────────────────────────────────────────
    print("\n[2/3] Mô phỏng trạng thái nguy kịch...")
    configs_crit = [
        ("CRIT_SEVERE_FALL",    10, 1000),
        ("CRIT_HEATSTROKE",     15, 1500),
        ("CRIT_TOXIC_COLLAPSE", 10, 1000),
        ("CRIT_DESERT_HEAT",    12, 1500),
    ]
    for name, n_ev, n_rows in configs_crit:
        rows = simulate_user_timeline(name, n_ev, n_rows)
        all_data.extend(rows)
        summary[name] = {"num_events": n_ev, "rows_per_event": n_rows, "total_rows": len(rows)}
        print(f"    {name}: {len(rows):,} dòng")

    # ── Nhóm Cross-scenario ───────────────────────────────────────────────────
    print("\n[3/3] Mô phỏng kịch bản chuyển tiếp (cross-scenario)...")
    configs_cross = [
        ("NORMAL_WALKING",        "WARN_DEHYDRATION",    12, 750, "CROSS_WALK_DEHYDRATION"),
        ("NORMAL_HEAVY_CLIMB",    "WARN_INTERNAL_INJURY",12, 750, "CROSS_CLIMB_INJURY"),
        ("WARN_SLEEP_HYPOTHERMIA","CRIT_SEVERE_FALL",    12, 750, "CROSS_HYPOTHERMIA_FALL"),
        ("CRIT_DESERT_HEAT",      "CRIT_HEATSTROKE",     10, 750, "CROSS_DESERT_HEATSTROKE"),
        ("NORMAL_WALKING",        "CRIT_TOXIC_COLLAPSE", 12, 750, "CROSS_WALK_TOXIC"),
        ("WARN_DEHYDRATION",      "CRIT_HEATSTROKE",     12, 750, "CROSS_DEHYDRATION_HEATSTROKE"),
    ]
    for a, b, n_ev, n_rows, cross_key in configs_cross:
        rows = simulate_composite_scenario(a, b, n_ev, n_rows)
        all_data.extend(rows)
        summary[cross_key] = {
            "num_events": n_ev, "rows_per_event": n_rows,
            "total_rows": len(rows), "scenario_a": a, "scenario_b": b,
        }
        print(f"    {a} → {b}: {len(rows):,} dòng")

    # ── Tổng hợp & xuất ───────────────────────────────────────────────────────
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "user_id", "hr", "hrv", "body_temp", "env_temp",
        "humidity", "pressure", "oxygen", "uv_index", "aqi",
        "accel_mag", "fall_duration",
        "base_hr", "base_hrv",   # baseline cá nhân
        "label",
    ])

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Đã xuất dataset: {OUTPUT_CSV}")
    print(f"📊 Tổng số dòng : {len(df):,}")
    print(f"⏱  Tổng thời gian mô phỏng: {len(df) * TIME_STEP_S / 3600:.1f} giờ")
    print("📊 Phân bổ nhãn:")
    print(df["label"].value_counts().sort_index().to_string())

    _write_scenarios_json(summary)
