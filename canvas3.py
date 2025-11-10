# canvas_safe_fixed_clean_v2.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json

# ==========================================================
# Hilfsfunktionen
# ==========================================================

def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    """Entfernt Punkte, die sich zu nahe sind (keine Duplikate/Geister)."""
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    """Robuste findContours-Variante (kompatibel mit OpenCV 3/4)."""
    m = mask.copy()
    res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if isinstance(res, tuple) and len(res) == 2:
        contours = res[0]
    elif isinstance(res, tuple) and len(res) == 3:
        contours = res[1]
    else:
        contours = res[1] if len(res) > 1 else []

    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0):
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
                centers.append((cx, cy))
    return centers

def circular_mean_deg(h):
    """Berechnet kreisförmigen Mittelwert für Hue (0..180)."""
    angles = (h.astype(float) / 180.0) * 2.0 * np.pi
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    ang = np.arctan2(s, c)
    if ang < 0:
        ang += 2.0 * np.pi
    return (ang / (2.0 * np.pi)) * 180.0

def compute_hsv_range(points, hsv_img, radius=5):
    """Berechne robusten HSV-Bereich aus markierten Punkten."""
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

    # Medianwerte (robuster gegen Ausreißer)
    h_med = circular_mean_deg(h)
    s_med = float(np.median(s))
    v_med = float(np.median(v))

    n_points = len(points)
    tol_h = int(min(25, 10 + n_points * 3))
    tol_s = int(min(80, 30 + n_points * 10))
    tol_v = int(min(80, 30 + n_points * 10))

    if np.std(h) > 25:
        tol_h = min(40, tol_h + 5)

    h_min = int(round((h_med - tol_h) % 180))
    h_max = int(round((h_med + tol_h) % 180))
    s_min = max(0, int(round(s_med - tol_s)))
    s_max = min(255, int(round(s_med + tol_s)))
    v_min = max(0, int(round(v_med - tol_v)))
    v_max = min(255, int(round(v_med + tol_v)))

    return (h_min, h_max, s_min, s_max, v_min, v_max)

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    """Maskiert unter Berücksichtigung von Hue-Wrap-around."""
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

def save_last_calibration():
    """Speichert HSV-Werte in JSON."""
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
        with open("kalibrierung.json", "w") as f:
            json.dump(data, f)
        st.success("💾 Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")

def load_last_calibration():
    """Lädt HSV-Kalibrierung aus JSON."""
    try:
        with open("kalibrierung.json", "r") as f:
            data = json.load(f)
        st.session_state.aec_hsv = np.array(data.get("aec_hsv")) if data.get("aec_hsv") else None
        st.session_state.hema_hsv = np.array(data.get("hema_hsv")) if data.get("hema_hsv") else None
        st.session_state.bg_hsv = np.array(data.get("bg_hsv")) if data.get("bg_hsv") else None
        st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Kalibrierung: {e}")

# ==========================================================
# Streamlit Setup
# ==========================================================
st.set_page_config(page_title="Zellkern-Zähler (v2)", layout="wide")
st.title("🧬 Zellkern-Zähler – 3-Punkt-Kalibrierung (stabil)")

# ----------------------------------------------------------
# Session State
# ----------------------------------------------------------
default_keys = [
    "aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema",
    "aec_hsv", "hema_hsv", "bg_hsv", "last_file", "disp_width", "last_auto_run"
]
for key in default_keys:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        elif key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = None

# ----------------------------------------------------------
# File upload
# ----------------------------------------------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset bei neuem Bild
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv", "hema_hsv", "bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# ----------------------------------------------------------
# Bild vorbereiten
# ----------------------------------------------------------
colW1, colW2 = st.columns([2, 1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# ==========================================================
# Sidebar: Parameter
# ==========================================================
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 2000, 100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius (Pixel)", 1, 15, 5)

st.sidebar.markdown("### 🎨 Modus auswählen")
mode = st.sidebar.radio("Modus", [
    "Keine",
    "AEC markieren (Kalibrierung)",
    "Hämatoxylin markieren (Kalibrierung)",
    "Hintergrund markieren",
    "AEC manuell hinzufügen",
    "Hämatoxylin manuell hinzufügen",
    "Punkt löschen (alle Kategorien)"
])

aec_mode = mode == "AEC markieren (Kalibrierung)"
hema_mode = mode == "Hämatoxylin markieren (Kalibrierung)"
bg_mode = mode == "Hintergrund markieren"
manual_aec_mode = mode == "AEC manuell hinzufügen"
manual_hema_mode = mode == "Hämatoxylin manuell hinzufügen"
delete_mode = mode == "Punkt löschen (alle Kategorien)"

# Schnellaktionen
st.sidebar.markdown("### ⚡ Schnellaktionen")
if st.sidebar.button("🧹 Alle Punkte löschen"):
    for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    st.success("Alle Punkte gelöscht.")

if st.sidebar.button("🧾 Kalibrierung zurücksetzen"):
    st.session_state.aec_hsv = None
    st.session_state.hema_hsv = None
    st.session_state.bg_hsv = None
    st.info("Kalibrierungswerte zurückgesetzt.")

if st.sidebar.button("🤖 Auto-Erkennung ausführen"):
    st.session_state.last_auto_run += 1

# Speichern / Laden
st.sidebar.markdown("### 💾 Kalibrierung")
if st.sidebar.button("💾 Speichern"):
    save_last_calibration()
if st.sidebar.button("📂 Laden"):
    load_last_calibration()

# ==========================================================
# Bildanzeige mit Markierungen
# ==========================================================
marked_disp = image_disp.copy()
for points_list, color in [
    (st.session_state.aec_points, (255, 100, 100)),
    (st.session_state.hema_points, (100, 100, 255)),
    (st.session_state.bg_points, (255, 255, 0)),
    (st.session_state.manual_aec, (255, 165, 0)),
    (st.session_state.manual_hema, (128, 0, 128))
]:
    for (x, y) in points_list:
        cv2.circle(marked_disp, (x, y), circle_radius, color, -1)

image_key = f"clickable_image_{st.session_state.last_auto_run}_{uploaded_file.name}"
coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=image_key, width=DISPLAY_WIDTH)

if coords and "x" in coords and "y" in coords:
    try:
        x = int(round(float(coords["x"])))
        y = int(round(float(coords["y"])))
    except Exception:
        x, y = None, None

    if x is not None and y is not None and 0 <= x < DISPLAY_WIDTH and 0 <= y < marked_disp.shape[0]:
        if delete_mode:
            for key in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
                st.session_state[key] = [p for p in st.session_state[key] if not is_near(p, (x, y), circle_radius)]
        elif aec_mode:
            st.session_state.aec_points.append((x, y))
        elif hema_mode:
            st.session_state.hema_points.append((x, y))
        elif bg_mode:
            st.session_state.bg_points.append((x, y))
        elif manual_aec_mode:
            st.session_state.manual_aec.append((x, y))
        elif manual_hema_mode:
            st.session_state.manual_hema.append((x, y))

for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# ==========================================================
# Kalibrierung
# ==========================================================
st.markdown("### ⚙️ Kalibrierung")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⚡ AEC kalibrieren"):
        if st.session_state.aec_points:
            st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
            st.session_state.aec_points = []
            st.success("✅ AEC-Kalibrierung gespeichert.")
        else:
            st.warning("Keine Punkte.")
with col2:
    if st.button("⚡ Hämatoxylin kalibrieren"):
        if st.session_state.hema_points:
            st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
            st.session_state.hema_points = []
            st.success("✅ Hämatoxylin-Kalibrierung gespeichert.")
        else:
            st.warning("Keine Punkte.")
with col3:
    if st.button("⚡ Hintergrund kalibrieren"):
        if st.session_state.bg_points:
            st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
            st.session_state.bg_points = []
            st.success("✅ Hintergrund-Kalibrierung gespeichert.")
        else:
            st.warning("Keine Punkte.")

# ==========================================================
# Auto-Erkennung
# ==========================================================
if st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    aec_detected, hema_detected = [], []

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

    aec_detected = get_centers(mask_aec, int(min_area))
    hema_detected = get_centers(mask_hema, int(min_area))

    if st.session_state.bg_points:
        aec_detected = remove_near(aec_detected, st.session_state.bg_points, r=max(6, circle_radius))
        hema_detected = remove_near(hema_detected, st.session_state.bg_points, r=max(6, circle_radius))

    merged_aec = list(st.session_state.manual_aec)
    for p in aec_detected:
        if not any(is_near(p, q, r=max(6, circle_radius)) for q in merged_aec):
            merged_aec.append(p)
    merged_hema = list(st.session_state.manual_hema)
    for p in hema_detected:
        if not any(is_near(p, q, r=max(6, circle_radius)) for q in merged_hema):
            merged_hema.append(p)

    st.session_state.aec_points = dedup_points(merged_aec)
    st.session_state.hema_points = dedup_points(merged_hema)
    st.session_state.last_auto_run = 0

# ==========================================================
# Ergebnisse & Export
# ==========================================================
all_aec = st.session_state.aec_points or []
all_hema = st.session_state.hema_points or []

n_aec = len(all_aec)
n_hema = len(all_hema)

st.markdown("### 📊 Ergebnisse")
colA, colB = st.columns(2)
with colA:
    st.metric("AEC-positive Zellen", n_aec)
with colB:
    st.metric("Hämatoxylin-positive Zellen", n_hema)

# Vorschau-Markierung im Bild
result_img = image_disp.copy()
for (x, y) in all_aec:
    cv2.circle(result_img, (x, y), circle_radius, (255, 0, 0), 2)
for (x, y) in all_hema:
    cv2.circle(result_img, (x, y), circle_radius, (0, 0, 255), 2)

st.image(result_img, caption=f"Erkannte Zellen (AEC={n_aec}, Häma={n_hema})", use_container_width=True)

# Exportoptionen
csv_data = pd.DataFrame({
    "Typ": ["AEC"] * n_aec + ["Hämatoxylin"] * n_hema,
    "x": [p[0] for p in all_aec + all_hema],
    "y": [p[1] for p in all_aec + all_hema],
})
csv_bytes = csv_data.to_csv(index=False).encode("utf-8")
st.download_button("📥 Ergebnisse als CSV herunterladen", csv_bytes, "zell_ergebnisse.csv", "text/csv")
