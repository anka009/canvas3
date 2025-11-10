import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# OpenCV -> PIL konvertieren
bg_image = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
bg_image = Image.fromarray(bg_image)

st.markdown("### 🖱️ Klick ins Bild, um Punkte zu setzen oder zu löschen")

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=0,
    background_image=bg_image,   # <--- hier jetzt PIL-Objekt
    update_streamlit=True,
    height=disp_img.shape[0],
    width=disp_img.shape[1],
    drawing_mode="point",
    key="canvas_click"
)

# ==========================================================
# Hilfsfunktionen
# ==========================================================
def compute_hsv_range(points, hsv_img, tol=15):
    hsv_vals = [hsv_img[y, x] for x, y in points if 0 <= x < hsv_img.shape[1] and 0 <= y < hsv_img.shape[0]]
    if not hsv_vals:
        return None
    hsv_vals = np.array(hsv_vals)
    mean = np.mean(hsv_vals, axis=0)
    lower = np.clip(mean - tol, 0, 255)
    upper = np.clip(mean + tol, 0, 255)
    return (lower, upper)

def is_near(p1, p2, radius):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < radius

def dedup_points(points, min_dist=8):
    if not points:
        return []
    filtered = []
    for p in points:
        if not any(is_near(p, q, min_dist) for q in filtered):
            filtered.append(p)
    return filtered

def mark_points(image, points, color, size=6):
    for (x, y) in points:
        cv2.circle(image, (x, y), size, color, -1)
    return image

# ==========================================================
# Initialisierung
# ==========================================================
st.set_page_config(layout="wide")
st.title("🔬 Farbpunkt-Analyse mit manueller Korrektur")

if "img" not in st.session_state:
    st.session_state.update({
        "aec_points": [], "hema_points": [], "bg_points": [],
        "manual_aec": [], "manual_hema": [],
        "aec_hsv": None, "hema_hsv": None, "bg_hsv": None,
        "fixed_auto": False
    })

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.header("⚙️ Einstellungen")
point_size = st.sidebar.slider("Punktgröße", 4, 20, 8, 1)
cluster_dist = st.sidebar.slider("Abstand zum Zusammenfassen (px)", 5, 50, 15, 1)
min_auto_points = st.sidebar.slider("Minimale Punkte für Auto-Kalibrierung", 5, 200, 50, 5)
st.sidebar.info("💡 Nach Kalibrierung wird automatisch ein Ergebnisbild erstellt.")

# ==========================================================
# Bild laden
# ==========================================================
uploaded = st.sidebar.file_uploader("Bild hochladen", type=["jpg", "png", "tif", "tiff"])
if not uploaded:
    st.stop()

file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
disp_img = img.copy()

# ==========================================================
# Modusauswahl
# ==========================================================
mode = st.radio("Modus wählen:", [
    "AEC kalibrieren", "Hämatoxylin kalibrieren", "Hintergrund kalibrieren",
    "Manuell AEC hinzufügen", "Manuell Hämatoxylin hinzufügen", "Punkte löschen"
])

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🧹 Alles löschen"):
        for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
            st.session_state[k].clear()
        for k in ["aec_hsv", "hema_hsv", "bg_hsv"]:
            st.session_state[k] = None
        st.session_state.fixed_auto = False
        st.experimental_rerun()

# ==========================================================
# Klickverarbeitung
# ==========================================================
from streamlit_drawable_canvas import st_canvas

st.markdown("### 🖱️ Klick ins Bild, um Punkte zu setzen oder zu löschen")

canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=0,
    background_image=cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB),
    update_streamlit=True,
    height=disp_img.shape[0],
    width=disp_img.shape[1],
    drawing_mode="point",
    key="canvas_click"
)

if canvas_result.json_data is not None:
    for obj in canvas_result.json_data["objects"]:
        x, y = int(obj["left"]), int(obj["top"])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if mode == "AEC kalibrieren":
                st.session_state.aec_points.append((x, y))
            elif mode == "Hämatoxylin kalibrieren":
                st.session_state.hema_points.append((x, y))
            elif mode == "Hintergrund kalibrieren":
                st.session_state.bg_points.append((x, y))
            elif mode == "Manuell AEC hinzufügen":
                st.session_state.manual_aec.append((x, y))
            elif mode == "Manuell Hämatoxylin hinzufügen":
                st.session_state.manual_hema.append((x, y))
            elif mode == "Punkte löschen":
                for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
                    st.session_state[k] = [p for p in st.session_state[k] if not is_near(p, (x, y), point_size + 2)]

# ==========================================================
# Automatische Erkennung nach Kalibrierung
# ==========================================================
if all(v is not None for v in [st.session_state.aec_hsv, st.session_state.hema_hsv, st.session_state.bg_hsv]) and not st.session_state.fixed_auto:
    # Masken erstellen
    mask_aec = cv2.inRange(hsv_img, st.session_state.aec_hsv[0], st.session_state.aec_hsv[1])
    mask_hema = cv2.inRange(hsv_img, st.session_state.hema_hsv[0], st.session_state.hema_hsv[1])

    # Konturen finden
    def extract_points(mask, min_area=10):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = [tuple(np.mean(c.reshape(-1, 2), axis=0).astype(int)) for c in contours if cv2.contourArea(c) > min_area]
        return dedup_points(pts, min_dist=cluster_dist)

    st.session_state.aec_points = extract_points(mask_aec)
    st.session_state.hema_points = extract_points(mask_hema)
    st.session_state.fixed_auto = True
    st.success("✅ Automatische Erkennung abgeschlossen.")

# ==========================================================
# Darstellung
# ==========================================================
out = img.copy()
mark_points(out, st.session_state.aec_points, (0, 255, 255), size=point_size)
mark_points(out, st.session_state.hema_points, (255, 0, 255), size=point_size)
mark_points(out, st.session_state.bg_points, (255, 255, 0), size=point_size)
mark_points(out, st.session_state.manual_aec, (0, 128, 255), size=point_size)
mark_points(out, st.session_state.manual_hema, (255, 128, 0), size=point_size)

st.image(out, channels="BGR", caption="Analysebild", use_container_width=True)

# ==========================================================
# Ergebnis anzeigen
# ==========================================================
st.markdown("### 🧮 Ergebnisse")
st.write(f"**AEC auto:** {len(st.session_state.aec_points)}")
st.write(f"**Hämatoxylin auto:** {len(st.session_state.hema_points)}")
st.write(f"**AEC manuell:** {len(st.session_state.manual_aec)}")
st.write(f"**Hämatoxylin manuell:** {len(st.session_state.manual_hema)}")
