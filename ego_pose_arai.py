#!/usr/bin/env python3
 
import os
import pandas as pd
import numpy as np
 
# ======== CONFIGURE ========
 
ACCEL_CSV = "/home/miteshv/Downloads/ARAI/IMU_Data/imu_acceleration.csv"  # ax, ay, (optional az), time
GYRO_CSV  = "/home/miteshv/Downloads/ARAI/IMU_Data/imu_gyros.csv"          # z gyroscope, timestamp
OUTPUT_CSV = "/home/miteshv/Documents/data_converter/Arai/your_file_with_pose.csv"
 
TIME_UNITS  = "us"      # "us" (microseconds), "ms", "s"
GYRO_UNITS  = "deg/s"   # "rad/s" or "deg/s"
SMOOTH_WINDOW_SEC = 0.30
BIAS_WINDOW_SEC   = 0.0
INITIAL_HEADING_DEG = 0.0
 
# ======== HELPERS ========
 
def sniff_delimiter(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    commas = header.count(",")
    semis  = header.count(";")
    return ";" if semis > commas else ","
 
def read_csv_flex(path):
    sep = sniff_delimiter(path)
    df = pd.read_csv(path, sep=sep, engine="python")
 
    if df.shape[1] == 1 and ("," in df.columns[0] or ";" in df.columns[0]):
        raw = pd.read_csv(path, header=None, sep="\n", engine="python")
        head = raw.iloc[0, 0]
        body = raw.iloc[1:, 0]
        delim = "," if head.count(",") >= head.count(";") else ";"
        cols = [c.strip() for c in head.split(delim)]
        rows = [[c.strip() for c in line.split(delim)] for line in body]
        df = pd.DataFrame(rows, columns=cols)
    return df
 
def norm_cols(df):
    return df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
 
def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None
 
def ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
 
def rolling_mean(x, N):
    s = pd.Series(x)
    return s.rolling(window=N, center=True, min_periods=1).mean().to_numpy()
 
def cumtrapz_manual(y, t):
    out = np.zeros_like(y, dtype=float)
    for i in range(1, len(y)):
        dt = t[i] - t[i-1]
        out[i] = out[i-1] + 0.5 * (y[i] + y[i-1]) * dt
    return out
 
# ======== LOAD ACCEL ========
 
acc = read_csv_flex(ACCEL_CSV)
acc = norm_cols(acc)
 
time_a = pick(acc, "time", "timestamp", "time_us", "t")
ax_c   = pick(acc, "ax", "a_x", "accel_x", "acceleration_x")
ay_c   = pick(acc, "ay", "a_y", "accel_y", "acceleration_y")
az_c   = pick(acc, "z_accelerometer", "az", "accel_z", "acceleration_z", "z")
 
if time_a is None or ax_c is None or ay_c is None:
    raise ValueError(f"Accel CSV missing required columns. Found: {list(acc.columns)}")
 
acc = ensure_numeric(acc, [time_a, ax_c, ay_c] + ([az_c] if az_c else []))
acc = acc.dropna(subset=[time_a, ax_c, ay_c]).sort_values(time_a).reset_index(drop=True)
 
t_a_raw = acc[time_a].to_numpy(dtype=float)
 
if TIME_UNITS == "us":
    t_a = t_a_raw * 1e-6
elif TIME_UNITS == "ms":
    t_a = t_a_raw * 1e-3
else:
    t_a = t_a_raw * 1.0
 
t_a -= t_a[0]
 
ax_b = acc[ax_c].to_numpy(dtype=float)
ay_b = acc[ay_c].to_numpy(dtype=float)
 
# ======== LOAD GYRO ========
 
gyro = read_csv_flex(GYRO_CSV)
gyro = norm_cols(gyro)
 
time_g = pick(gyro, "timestamp", "time", "time_us", "t")
gz_c   = pick(gyro, "z_gyroscope", "gz", "gyro_z", "yaw_rate", "z")
 
if time_g is None or gz_c is None:
    raise ValueError(f"Gyro CSV missing required columns. Found: {list(gyro.columns)}")
 
gyro = ensure_numeric(gyro, [time_g, gz_c])
gyro = gyro.dropna(subset=[time_g, gz_c]).sort_values(time_g).reset_index(drop=True)
 
t_g_raw = gyro[time_g].to_numpy(dtype=float)
 
if TIME_UNITS == "us":
    t_g = t_g_raw * 1e-6
elif TIME_UNITS == "ms":
    t_g = t_g_raw * 1e-3
else:
    t_g = t_g_raw * 1.0
 
t0 = min(t_a[0], t_g[0])
t_a = t_a - t0
t_g = t_g - t0
 
gz = gyro[gz_c].to_numpy(dtype=float)
if GYRO_UNITS.lower() in ["deg/s", "degps", "dps"]:
    gz = np.deg2rad(gz)
 
# ======== INTERPOLATE gz ONTO ACCEL TIMELINE ========
 
gz_i = np.interp(t_a, t_g, gz, left=gz[0], right=gz[-1])
 
# ======== SMOOTHING ========
 
median_dt = float(np.median(np.diff(t_a))) if len(t_a) > 1 else 0.01
N = max(1, int(round(SMOOTH_WINDOW_SEC / median_dt))) if median_dt > 0 else 5
 
ax_b_f = rolling_mean(ax_b, N)
ay_b_f = rolling_mean(ay_b, N)
gz_f   = rolling_mean(gz_i, N)
 
# ======== OPTIONAL BIAS REMOVAL ========
 
if BIAS_WINDOW_SEC > 0:
    idx = np.searchsorted(t_a, BIAS_WINDOW_SEC, side="right")
    idx = max(1, min(idx, len(t_a)))
    ax_b_f -= float(np.mean(ax_b_f[:idx]))
    ay_b_f -= float(np.mean(ay_b_f[:idx]))
    gz_f   -= float(np.mean(gz_f[:idx]))
 
# ======== INTEGRATE YAW → HEADING ========
 
theta = cumtrapz_manual(gz_f, t_a)
theta += np.deg2rad(INITIAL_HEADING_DEG)
 
# ======== ROTATE BODY → WORLD ========
 
c, s = np.cos(theta), np.sin(theta)
ax_w = ax_b_f * c - ay_b_f * s
ay_w = ax_b_f * s + ay_b_f * c
 
# ======== INTEGRATE TO VELOCITIES & POSITIONS ========
 
vx = cumtrapz_manual(ax_w, t_a)
vy = cumtrapz_manual(ay_w, t_a)
 
x = cumtrapz_manual(vx, t_a)
y = cumtrapz_manual(vy, t_a)
 
speed = np.hypot(vx, vy)
 
# ======== COMPUTE DISTANCE & DISPLACEMENT ========
 
dist_from_start = np.hypot(x, y)
 
path_length = np.zeros_like(speed)
for i in range(1, len(path_length)):
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    step_dist = np.hypot(dx, dy)
    path_length[i] = path_length[i-1] + step_dist
 
total_distance = float(path_length[-1]) if len(path_length) else 0.0
final_displacement = float(dist_from_start[-1]) if len(dist_from_start) else 0.0
 
# ======== SAVE OUTPUT ========
 
out = pd.DataFrame({
    "time_s": t_a,
    "vx_mps": vx,
    "vy_mps": vy,
    "speed_mps": speed,
    "x_m": x,
    "y_m": y,
    "dist_from_start_m": dist_from_start,
    "path_length_m": path_length
})
 
out.to_csv(OUTPUT_CSV, index=False)
 
# ======== SUMMARY ========
 
duration = float(t_a[-1]) if len(t_a) else 0.0
 
print(f"Saved: {OUTPUT_CSV}")
print(f"samples={len(t_a)}, duration={duration:.3f}s, median_dt={median_dt:.6f}s")
print(f"Final displacement: {final_displacement:.2f} m")
print(f"Total path distance: {total_distance:.2f} m")
print(f"Final speed: {speed[-1]:.2f} m/s")