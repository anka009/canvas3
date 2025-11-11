# canvas2_auto_calib.py
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
    """Erweiterte Konturenerkennung mit Glättung und Morphologie."""
    # Morphologische Filterung
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Konturen finden
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
    """Robuste Median-basierte HSV-Range-Berechnung mit Wrap-Achtung für Hue."""
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

    # Medianwerte
    h_med = float(np.median(h))
    s_med = float(np.median(s))
    v_med = float(np.median(v))

    n_points = len(points)
    tol_h = int(min(25, 10 + n_points * 3))
    tol_s = int(min(80, 30 + n_points * 10))
    tol_v = int(min(80, 30 + n_points * 10))

    # rudimentärer Hue-Wrap-Aware-Fix: falls Werte um 0/179 gruppiert sind
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
st.set_page_config(page_title="Zellkern-Zähler (Auto-Kalib)", layout="wide")
st.title("🧬 Zellkern-Zähler – Auto-Kalibrierung (AEC / Hämatoxylin)")

# -------------------- Session State: neue, saubere Struktur --------------------
default_lists = [
    "aec_cal_points", "hema_cal_points", "bg_cal_points",   # temporäre Kalibrier-Punkte
    "aec_auto", "hema_auto",                               # automatische Ergebnisse
    "manual_aec", "manual_hema",                           # manuelle Punkte
    "aec_hsv", "hema_hsv", "bg_hsv",                       # gespeicherte HSV-Kalibrierungen
    "last_file", "disp_width", "last_auto_run"
]
for key in default_lists:
    if key not in st.session_state:
        if key in ["aec_hsv", "hema_hsv", "bg_hsv"]:
            st.session_state[key] = None
        elif key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = []

# 👉 Neue, gezielte "ersten Klick ignorieren"-Flags pro Kategorie
for flag in ["aec_first_ignore", "hema_first_ignore", "bg_first_ignore"]:
    if flag not in st.session_state:
        st.session_state[flag] = True  # beim Start: ersten Klick ignorieren

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset state on new file by name (if you want stronger check, use hash of bytes)
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "aec_auto", "hema_auto", "manual_aec", "manual_hema"]:
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

# load image
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar: Parameter --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 2000, 100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius (Pixel)", 1, 15, 5)

min_points_calib = st.sidebar.slider(
    "🧮 Minimale Punkte für automatische Kalibrierung",
    min_value=1, max_value=10, value=3, step=1
)
st.sidebar.info("Kalibrierung läuft automatisch, sobald die minimale Punktzahl erreicht ist.")

# -------------------- Bildanzeige mit Markierungen --------------------
marked_disp = image_disp.copy()
# draw calibration input points (small filled)
for (x, y) in st.session_state.aec_cal_points:
    cv2.circle(marked_disp, (x, y), max(2, circle_radius), (0, 120, 200), -1)  # teal-ish - aec cal
for (x, y) in st.session_state.hema_cal_points:
    cv2.circle(marked_disp, (x, y), max(2, circle_radius), (200, 120, 0), -1)  # orange-ish - hema cal
for (x, y) in st.session_state.bg_cal_points:
    cv2.circle(marked_disp, (x, y), max(2, circle_radius), (200, 200, 0), -1)  # yellow - bg cal

# draw manual points (filled)
for (x, y) in st.session_state.manual_aec:
    cv2.circle(marked_disp, (x, y), circle_radius, (0, 165, 255), -1)  # orange filled = manual aec
for (x, y) in st.session_state.manual_hema:
    cv2.circle(marked_disp, (x, y), circle_radius, (128, 0, 128), -1)  # purple filled = manual hema

# draw auto-detected points (outlined)
for (x, y) in st.session_state.aec_auto:
    cv2.circle(marked_disp, (x, y), circle_radius, (0, 0, 255), 2)  # red outline = aec auto
for (x, y) in st.session_state.hema_auto:
    cv2.circle(marked_disp, (x, y), circle_radius, (255, 0, 0), 2)  # blue outline = hema auto

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=f"clickable_image_{st.session_state.last_auto_run}_{st.session_state.last_file}", width=DISPLAY_WIDTH)

# --- Klick-Logik ---
if click_coords:
    if st.session_state.mode == "AEC Kalibrier-Punkt setzen":
        st.session_state.aec_cal_points.append(click_coords)

    elif st.session_state.mode == "Hämatoxylin Kalibrier-Punkt setzen":
        st.session_state.hema_cal_points.append(click_coords)

    elif st.session_state.mode == "Hintergrund Kalibrier-Punkt setzen":
        st.session_state.bg_cal_points.append(click_coords)

    elif st.session_state.mode == "AEC manuell hinzufügen":
        st.session_state.manual_aec.append(click_coords)

    elif st.session_state.mode == "Hämatoxylin manuell hinzufügen":
        st.session_state.manual_hema.append(click_coords)

    elif st.session_state.mode == "Punkt löschen":
        # Beispiel: Löschlogik für alle Listen
        for lst in [
            st.session_state.aec_cal_points,
            st.session_state.hema_cal_points,
            st.session_state.bg_cal_points,
            st.session_state.aec_auto,
            st.session_state.hema_auto,
            st.session_state.manual_aec,
            st.session_state.manual_hema,
        ]:
            lst[:] = [p for p in lst if np.linalg.norm(np.array(p) - np.array(click_coords)) > circle_radius]

# Deduplication
for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# Auto-Kalibrierung aus Kalibrier-Punkten
def auto_calibrate_from_calpoints(category_name, cal_key, hsv_key, hsv_img, radius):
    pts = st.session_state.get(cal_key, [])
    if len(pts) >= min_points_calib:
        hsv = compute_hsv_range(pts, hsv_img, radius=radius)
        if hsv is not None:
            st.session_state[hsv_key] = hsv
            st.success(f"✅ {category_name}: Kalibrierung automatisch ({len(pts)} Punkte)")
            # Reset cal points (they were just used)
            st.session_state[cal_key] = []
            st.session_state.last_auto_run += 1

# Background: keep an option to keep points or clear - here we clear after computing bg_hsv
if len(st.session_state.bg_cal_points) >= min_points_calib:
    hsv_bg = compute_hsv_range(st.session_state.bg_cal_points, hsv_disp, radius=calib_radius)
    if hsv_bg is not None:
        st.session_state.bg_hsv = hsv_bg
        st.success(f"✅ Hintergrund-Kalibrierung automatisch ({len(st.session_state.bg_cal_points)} Punkte)")
        st.session_state.bg_cal_points = []  # we clear them because bg_hsv is stored
        st.session_state.last_auto_run += 1

auto_calibrate_from_calpoints("AEC", "aec_cal_points", "aec_hsv", hsv_disp, calib_radius)
auto_calibrate_from_calpoints("Hämatoxylin", "hema_cal_points", "hema_hsv", hsv_disp, calib_radius)

# -------------------- Auto-Erkennung (wird auf last_auto_run > 0 getriggert) --------------------
if st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    if st.session_state.aec_hsv is not None:
        mask_aec = apply_hue_wrap(hsv_proc, *map(int, st.session_state.aec_hsv))
    else:
        mask_aec = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    if st.session_state.hema_hsv is not None:
        mask_hema = apply_hue_wrap(hsv_proc, *map(int, st.session_state.hema_hsv))
    else:
        mask_hema = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    if st.session_state.bg_hsv is not None:
        mask_bg = apply_hue_wrap(hsv_proc, *map(int, st.session_state.bg_hsv))
        mask_aec = cv2.bitwise_and(mask_aec, cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema, cv2.bitwise_not(mask_bg))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel)
    mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel)

    detected_aec = get_centers(mask_aec, int(min_area))
    detected_hema = get_centers(mask_hema, int(min_area))

    # Optionally remove detections near manually marked bg points (if you kept bg points elsewhere)
    # if st.session_state.get("bg_points"):
    #     detected_aec = remove_near(detected_aec, st.session_state.bg_points, r=max(6, circle_radius))
    #     detected_hema = remove_near(detected_hema, st.session_state.bg_points, r=max(6, circle_radius))

    # Save automatic detections separate from manual points
    st.session_state.aec_auto = dedup_points(detected_aec, min_dist=max(4, circle_radius // 2))
    st.session_state.hema_auto = dedup_points(detected_hema, min_dist=max(4, circle_radius // 2))

    st.session_state.last_auto_run = 0

# --- Manueller Modus (Batch-Logik) ---
if st.session_state.mode in ["AEC manuell hinzufügen", "Hämatoxylin manuell hinzufügen"]:
    # Basisbild einmal speichern (mit Auto-Markierungen)
    if "manual_base_image" not in st.session_state:
        st.session_state.manual_base_image = image_with_auto_marks.copy()

    # Marker sammeln (nur Koordinaten speichern, keine sofortige Berechnung)
    if click_coords:
        if st.session_state.mode == "AEC manuell hinzufügen":
            st.session_state.manual_aec.append(click_coords)
        elif st.session_state.mode == "Hämatoxylin manuell hinzufügen":
            st.session_state.manual_hema.append(click_coords)

    # Vorschau: Basisbild + Marker anzeigen
    preview = st.session_state.manual_base_image.copy()
    for (x, y) in st.session_state.manual_aec:
        cv2.circle(preview, (x, y), 5, (255, 0, 0), -1)  # Blau für AEC
    for (x, y) in st.session_state.manual_hema:
        cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)  # Rot für Hämatoxylin
    st.image(preview, caption="Manuelle Marker Vorschau")

    # Button zum Abschluss
    if st.button("Fertig"):
        st.session_state.manual_aec = dedup_points(
            st.session_state.manual_aec,
            min_dist=max(4, circle_radius // 2)
        )
        st.session_state.manual_hema = dedup_points(
            st.session_state.manual_hema,
            min_dist=max(4, circle_radius // 2)
        )
        st.session_state.manual_done = True
        st.success("✅ Manuelle Punkte übernommen und berechnet!")

# --- Ergebnis-Block ---
st.markdown("### 📊 Ergebnisse")
colA, colB = st.columns(2)

with colA:
    st.metric("AEC (auto)", len(st.session_state.aec_auto))
    if "manual_done" in st.session_state and st.session_state.manual_done:
        st.metric("AEC (manuell)", len(st.session_state.manual_aec))
    else:
        st.metric("AEC (manuell)", 0)

with colB:
    st.metric("Hämatoxylin (auto)", len(st.session_state.hema_auto))
    if "manual_done" in st.session_state and st.session_state.manual_done:
        st.metric("Hämatoxylin (manuell)", len(st.session_state.manual_hema))
    else:
        st.metric("Hämatoxylin (manuell)", 0)

# --- Debug-Block ---
with st.expander("🧠 Debug Info"):
    debug_info = {
        "aec_hsv": st.session_state.aec_hsv,
        "hema_hsv": st.session_state.hema_hsv,
        "bg_hsv": st.session_state.bg_hsv,
        "aec_auto_count": len(st.session_state.aec_auto),
        "hema_auto_count": len(st.session_state.hema_auto),
        "aec_cal_points": st.session_state.aec_cal_points,
        "hema_cal_points": st.session_state.hema_cal_points,
        "bg_cal_points": st.session_state.bg_cal_points,
        "last_auto_run": st.session_state.last_auto_run,
        "image_shape": image_disp.shape if isinstance(image_disp, np.ndarray) else None
    }

    if "manual_done" in st.session_state and st.session_state.manual_done:
        debug_info["manual_aec_count"] = len(st.session_state.manual_aec)
        debug_info["manual_hema_count"] = len(st.session_state.manual_hema)
    else:
        debug_info["manual_aec_count"] = 0
        debug_info["manual_hema_count"] = 0

    st.write(debug_info)

# --- CSV-Export ---
if st.button("📥 Export CSV"):
    export_data = {
        "AEC_auto": len(st.session_state.aec_auto),
        "Hema_auto": len(st.session_state.hema_auto),
    }

    if "manual_done" in st.session_state and st.session_state.manual_done:
        export_data["AEC_manual"] = len(st.session_state.manual_aec)
        export_data["Hema_manual"] = len(st.session_state.manual_hema)
    else:
        export_data["AEC_manual"] = 0
        export_data["Hema_manual"] = 0

    df_export = pd.DataFrame([export_data])
    st.dataframe(df_export)
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "results.csv", "text/csv")
