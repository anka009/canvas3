# canvas2_auto_calib_fast_manual_full.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        c = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(c)
        if area >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0):
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
                centers.append((cx, cy))
    return centers

def compute_hsv_range(points, hsv_img, radius=5):
    if not points:
        return None
    vals = []
    for (x, y) in points:
        x_min = max(0, x - radius)
        x_max = min(hsv_img.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(hsv_img.shape[0], y + radius + 1)
        region = hsv_img[y_min:y_max, x_min:x_max]
        if region.size > 0:
            vals.append(region.reshape(-1, 3))
    if not vals:
        return None
    vals = np.vstack(vals)
    h = vals[:, 0].astype(int)
    s = vals[:, 1].astype(int)
    v = vals[:, 2].astype(int)
    h_med = float(np.median(h))
    s_med = float(np.median(s))
    v_med = float(np.median(v))
    n_points = len(points)
    tol_h = int(min(25, 10 + n_points * 3))
    tol_s = int(min(80, 30 + n_points * 10))
    tol_v = int(min(80, 30 + n_points * 10))
    if np.mean(h) > 150 or np.mean(h) < 20:
        h_med = float(np.median(np.where(h < 90, h + 180, h)) % 180)
        tol_h = min(40, tol_h + 5)
    h_min = int(round((h_med - tol_h) % 180))
    h_max = int(round((h_med + tol_h) % 180))
    s_min = max(0, int(round(s_med - tol_s)))
    s_max = min(255, int(round(s_med + tol_s)))
    v_min = max(0, int(round(v_med - tol_v)))
    v_max = min(255, int(round(v_med + tol_v)))
    return (h_min, h_max, s_min, s_max, v_min, v_max)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

def ensure_odd(k):
    return k if k % 2 == 1 else k + 1

def remove_near(points, forbidden_points, r):
    if not forbidden_points:
        return points
    return [p for p in points if not any(is_near(p, q, r) for q in forbidden_points)]

def save_last_calibration(path="kalibrierung.json"):
    def safe_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, list):
            return val
        else:
            return None
    data = {
        "aec_hsv": safe_list(st.session_state.get("aec_hsv")),
        "hema_hsv": safe_list(st.session_state.get("hema_hsv")),
        "bg_hsv": safe_list(st.session_state.get("bg_hsv"))
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f)
        st.success("💾 Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")

def load_last_calibration(path="kalibrierung.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        st.session_state.aec_hsv = np.array(data.get("aec_hsv")) if data.get("aec_hsv") else None
        st.session_state.hema_hsv = np.array(data.get("hema_hsv")) if data.get("hema_hsv") else None
        st.session_state.bg_hsv = np.array(data.get("bg_hsv")) if data.get("bg_hsv") else None
        st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (Fast Manual)", layout="wide")
st.title("🧬 Zellkern-Zähler – Auto-Kalibrierung mit Schnellmodus")

# -------------------- Session State --------------------
default_lists = [
    "aec_cal_points", "hema_cal_points", "bg_cal_points",
    "aec_auto", "hema_auto",
    "manual_aec", "manual_hema",
    "aec_hsv", "hema_hsv", "bg_hsv",
    "last_file", "disp_width", "last_auto_run",
    "temp_manual_aec", "temp_manual_hema"
]
for key in default_lists:
    if key not in st.session_state:
        if key in ["aec_hsv", "hema_hsv", "bg_hsv"]:
            st.session_state[key] = None
        elif key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = []

for flag in ["aec_first_ignore", "hema_first_ignore", "bg_first_ignore"]:
    if flag not in st.session_state:
        st.session_state[flag] = True

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "aec_auto", "hema_auto",
              "manual_aec", "manual_hema", "temp_manual_aec", "temp_manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv", "hema_hsv", "bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
colW1, colW2 = st.columns([2, 1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 2000, 100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius (Pixel)", 1, 15, 5)
min_points_calib = st.sidebar.slider("🧮 Minimale Punkte für automatische Kalibrierung", 1, 10, 3)

st.sidebar.markdown("### 🎨 Modus auswählen")
mode = st.sidebar.radio(
    "Modus",
    [
        "AEC Kalibrier-Punkt setzen",
        "Hämatoxylin Kalibrier-Punkt setzen",
        "Hintergrund Kalibrier-Punkt setzen",
        "AEC manuell hinzufügen",
        "Hämatoxylin manuell hinzufügen",
        "Punkt löschen"
    ],
    index=0
)

aec_mode = mode == "AEC Kalibrier-Punkt setzen"
hema_mode = mode == "Hämatoxylin Kalibrier-Punkt setzen"
bg_mode = mode == "Hintergrund Kalibrier-Punkt setzen"
manual_aec_mode = mode == "AEC manuell hinzufügen"
manual_hema_mode = mode == "Hämatoxylin manuell hinzufügen"
delete_mode = mode == "Punkt löschen"

if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = None
if mode != st.session_state.prev_mode:
    if "AEC" in mode:
        st.session_state.aec_first_ignore = True
    if "Hämatoxylin" in mode:
        st.session_state.hema_first_ignore = True
    if "Hintergrund" in mode:
        st.session_state.bg_first_ignore = True
    st.session_state.prev_mode = mode

if st.sidebar.button("🧹 Alle Punkte löschen"):
    for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "aec_auto", "hema_auto",
              "manual_aec", "manual_hema", "temp_manual_aec", "temp_manual_hema"]:
        st.session_state[k] = []
    st.success("Alle Punkte gelöscht.")

# -------------------- Bildanzeige --------------------
def draw_all_points():
    marked_disp = image_disp.copy()
    # Kalibrierpunkte
    for (x, y) in st.session_state.aec_cal_points:
        cv2.circle(marked_disp, (x, y), max(2, circle_radius), (0, 120, 200), -1)
    for (x, y) in st.session_state.hema_cal_points:
        cv2.circle(marked_disp, (x, y), max(2, circle_radius), (200, 120, 0), -1)
    for (x, y) in st.session_state.bg_cal_points:
        cv2.circle(marked_disp, (x, y), max(2, circle_radius), (200, 200, 0), -1)
    # Auto-Punkte
    for (x, y) in st.session_state.aec_auto:
        cv2.circle(marked_disp, (x, y), circle_radius, (0, 0, 255), 2)
    for (x, y) in st.session_state.hema_auto:
        cv2.circle(marked_disp, (x, y), circle_radius, (255, 0, 0), 2)
    # Manuell
    for (x, y) in st.session_state.manual_aec:
        cv2.circle(marked_disp, (x, y), circle_radius, (0, 165, 255), -1)
    for (x, y) in st.session_state.manual_hema:
        cv2.circle(marked_disp, (x, y), circle_radius, (128, 0, 128), -1)
    # Temporär
    for (x, y) in st.session_state.temp_manual_aec:
        cv2.circle(marked_disp, (x, y), circle_radius, (0, 200, 255), -1)
    for (x, y) in st.session_state.temp_manual_hema:
        cv2.circle(marked_disp, (x, y), circle_radius, (150, 0, 150), -1)
    return marked_disp

marked_disp = draw_all_points()

coords = streamlit_image_coordinates(Image.fromarray(marked_disp),
                                     key=f"clickable_image_{st.session_state.last_auto_run}_{st.session_state.last_file}",
                                     width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords:
    x, y = int(coords["x"]), int(coords["y"])
    if "first_click_ignored" not in st.session_state:
        st.session_state.first_click_ignored = False
    if not st.session_state.first_click_ignored:
        st.session_state.first_click_ignored = True
        st.info("⏳ Erster Klick ignoriert (Initialisierung).")
    else:
        if delete_mode:
            for key in ["aec_cal_points", "hema_cal_points", "bg_cal_points",
                        "manual_aec", "manual_hema", "temp_manual_aec", "temp_manual_hema"]:
                st.session_state[key] = [p for p in st.session_state[key] if not is_near(p, (x, y), circle_radius)]
            st.info("Punkt(e) gelöscht (falls gefunden).")
        elif aec_mode:
            if st.session_state.aec_first_ignore:
                st.session_state.aec_first_ignore = False
            else:
                st.session_state.aec_cal_points.append((x, y))
        elif hema_mode:
            if st.session_state.hema_first_ignore:
                st.session_state.hema_first_ignore = False
            else:
                st.session_state.hema_cal_points.append((x, y))
        elif bg_mode:
            if st.session_state.bg_first_ignore:
                st.session_state.bg_first_ignore = False
            else:
                st.session_state.bg_cal_points.append((x, y))
        elif manual_aec_mode:
            st.session_state.temp_manual_aec.append((x, y))
        elif manual_hema_mode:
            st.session_state.temp_manual_hema.append((x, y))

# Dedup
for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points",
          "manual_aec", "manual_hema", "temp_manual_aec", "temp_manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# -------------------- Buttons Fertig/Abbrechen --------------------
if manual_aec_mode or manual_hema_mode:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Fertig"):
            st.session_state.manual_aec.extend(st.session_state.temp_manual_aec)
            st.session_state.manual_hema.extend(st.session_state.temp_manual_hema)
            st.session_state.temp_manual_aec = []
            st.session_state.temp_manual_hema = []
            st.success("Manuelle Punkte übernommen.")
            st.session_state.last_auto_run += 1
    with col2:
        if st.button("❌ Abbrechen"):
            st.session_state.temp_manual_aec = []
            st.session_state.temp_manual_hema = []
            st.warning("Temporäre Punkte verworfen.")

# -------------------- Auto-Kalibrierung --------------------
def auto_calibrate_from_calpoints(category_name, cal_key, hsv_key, hsv_img, radius):
    pts = st.session_state.get(cal_key, [])
    if len(pts) >= min_points_calib:
        hsv = compute_hsv_range(pts, hsv_img, radius=radius)
        if hsv is not None:
            st.session_state[hsv_key] = hsv
            st.success(f"✅ {category_name}: Kalibrierung automatisch ({len(pts)} Punkte)")
            st.session_state[cal_key] = []
            st.session_state.last_auto_run += 1

if len(st.session_state.bg_cal_points) >= min_points_calib:
    hsv_bg = compute_hsv_range(st.session_state.bg_cal_points, hsv_disp, radius=calib_radius)
    if hsv_bg is not None:
        st.session_state.bg_hsv = hsv_bg
        st.success(f"✅ Hintergrund-Kalibrierung automatisch ({len(st.session_state.bg_cal_points)} Punkte)")
        st.session_state.bg_cal_points = []
        st.session_state.last_auto_run += 1

auto_calibrate_from_calpoints("AEC", "aec_cal_points", "aec_hsv", hsv_disp, calib_radius)
auto_calibrate_from_calpoints("Hämatoxylin", "hema_cal_points", "hema_hsv", hsv_disp, calib_radius)

# -------------------- Auto-Erkennung --------------------
if st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    mask_aec = apply_hue_wrap(hsv_proc, *(st.session_state.aec_hsv)) if st.session_state.aec_hsv else np.zeros_like(hsv_proc[:, :, 0])
    mask_hema = apply_hue_wrap(hsv_proc, *(st.session_state.hema_hsv)) if st.session_state.hema_hsv else np.zeros_like(hsv_proc[:, :, 0])

    st.session_state.aec_auto = get_centers(mask_aec, min_area=min_area)
    st.session_state.hema_auto = get_centers(mask_hema, min_area=min_area)
    st.session_state.last_auto_run += 1

# -------------------- Ergebnisanzeige --------------------
st.image(draw_all_points(), caption="Markierte Punkte", use_column_width=True)

# -------------------- CSV Export --------------------
def export_points_to_csv():
    df_list = []
    for name, points in [("AEC", st.session_state.aec_auto + st.session_state.manual_aec),
                         ("Hämatoxylin", st.session_state.hema_auto + st.session_state.manual_hema)]:
        for x, y in points:
            df_list.append({"Kategorie": name, "x": x, "y": y})
    if df_list:
        df = pd.DataFrame(df_list)
        csv_path = Path("punkte_export.csv")
        df.to_csv(csv_path, index=False)
        st.success(f"CSV exportiert: {csv_path}")
        st.download_button("⬇️ CSV herunterladen", df.to_csv(index=False), "punkte_export.csv", "text/csv")

export_points_to_csv()
