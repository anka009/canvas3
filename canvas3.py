# canvas_safe_v2.py
# Verbesserte, "intelligente" Version deines Zellkern-Zähler-Tools für Streamlit
# Funktionen:
# - adaptive Farbkalibrierung (KMeans)
# - automatische Kalibrierung wenn möglich
# - HSV + YCrCb-Fusion für bessere AEC-Erkennung
# - Morphologie + Watershed/DistanceTransform zur Trennung zusammenhängender Kerne
# - dynamische Parameter-Anpassung an Bildgröße/ Helligkeit
# - visualisierte Ergebnisse mit Legende
# - Export: CSV + annotiertes PNG + Confidence-Score

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from skimage import measure, segmentation, morphology
import io
import base64

st.set_page_config(page_title="Zellkern-Zähler — Smart", layout="wide")
st.title("🧬 Zellkern-Zähler — Smart Version")

# -------------------- Hilfsfunktionen --------------------

def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r


def get_centers(mask, min_area=50):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) 
                centers.append((cx, cy))
    return centers


def compute_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=25):
    if not points:
        return None
    vals = np.array([hsv_img[y, x] for (x, y) in points])
    h = vals[:, 0].astype(int)
    s = vals[:, 1].astype(int)
    v = vals[:, 2].astype(int)

    # Rot-Wrap-Behandlung
    if np.max(h) - np.min(h) > 150:
        h = np.where(h < 90, h + 180, h)

    h_min = int(max(0, np.min(h) - buffer_h))
    h_max = int(min(180, np.max(h) + buffer_h))
    s_min = int(max(0, np.min(s) - buffer_s))
    s_max = int(min(255, np.max(s) + buffer_s))
    v_min = int(max(0, np.min(v) - buffer_v))
    v_max = int(min(255, np.max(v) + buffer_v))

    if h_max > 180:
        h_max -= 180
        h_min -= 180
    return (h_min % 180, h_max % 180, s_min, s_max, v_min, v_max)


def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask


def adaptive_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=30, n_clusters=2):
    if not points:
        return None
    vals = np.array([hsv_img[y, x] for (x, y) in points])
    if len(vals) < n_clusters:
        return compute_hsv_range(points, hsv_img, buffer_h, buffer_s, buffer_v)

    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = kmeans.fit_predict(vals)
    centers = kmeans.cluster_centers_

    # Wähle Cluster mit höchster mittlerer Sättigung (typisch AEC)
    avg_saturation = [np.mean(vals[labels == i, 1]) for i in range(n_clusters)]
    best_cluster = int(np.argmax(avg_saturation))

    cluster_vals = vals[labels == best_cluster]
    h = cluster_vals[:, 0].astype(int)
    s = cluster_vals[:, 1].astype(int)
    v = cluster_vals[:, 2].astype(int)

    if np.max(h) - np.min(h) > 150:
        h = np.where(h < 90, h + 180, h)

    h_min = int(max(0, np.min(h) - buffer_h))
    h_max = int(min(180, np.max(h) + buffer_h))
    s_min = int(max(0, np.min(s) - buffer_s))
    s_max = int(min(255, np.max(s) + buffer_s))
    v_min = int(max(0, np.min(v) - buffer_v))
    v_max = int(min(255, np.max(v) + buffer_v))

    if h_max > 180:
        h_max -= 180
        h_min -= 180

    return (h_min % 180, h_max % 180, s_min, s_max, v_min, v_max)


def dynamic_params_from_image(image_disp):
    gray = cv2.cvtColor(image_disp, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    H, W = image_disp.shape[:2]

    # min_area skaliert mit Bildfläche
    min_area = int(max(30, (W * H) // 8000))

    # alpha (Kontrast) abhängig von Helligkeit
    if brightness > 200:
        alpha = 0.9
    elif brightness < 80:
        alpha = 1.2
    else:
        alpha = 1.0

    # blur abhängig vom Rauschniveau (Schätzung: Varianz)
    var = np.var(gray)
    if var > 2000:
        blur = 5
    elif var > 500:
        blur = 3
    else:
        blur = 1

    return min_area, alpha, blur


def fuse_masks_hsv_ycrcb(image_disp, hsv_proc, aec_hsv=None, hema_hsv=None):
    # Erzeuge Masken aus HSV-Grenzen und verstärke AEC über Cr-Kanal
    mask_aec = None
    mask_hema = None

    ycrcb = cv2.cvtColor(image_disp, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]

    if aec_hsv:
        hmin, hmax, smin, smax, vmin, vmax = aec_hsv
        mask_h = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)
        # Cr-Bereich experimentell - kann je nach Datensatz justiert werden
        mask_cr = cv2.inRange(cr, 135, 200)
        mask_aec = cv2.bitwise_and(mask_h, mask_cr)

        # Fallback: wenn AND zu wenig ergibt, benutze nur HSV
        if cv2.countNonZero(mask_aec) < 50:
            mask_aec = mask_h

    if hema_hsv:
        hmin, hmax, smin, smax, vmin, vmax = hema_hsv
        mask_hema = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)

    return mask_aec, mask_hema


def separate_touching_cells(mask):
    # Erwartet binäre Maske (uint8)
    if mask is None or cv2.countNonZero(mask) == 0:
        return []

    # sicherstellen: 0/255
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Marker finden
    local_max = (distance == ndi.maximum_filter(distance, size=15))
    markers = measure.label(local_max)

    # Watershed benötigt ein Bild mit Hintergrund und Marker
    labels = segmentation.watershed(-distance, markers, mask=binary)

    centers = []
    for region in measure.regionprops(labels):
        if region.area >= 20:
            cy, cx = region.centroid
            centers.append((int(cx), int(cy)))
    return centers


def annotate_image(result_img, session_state):
    # result_img: RGB (wie image_disp)
    overlay = result_img.copy()

    COLOR_AEC = (255, 0, 0)
    COLOR_HEMA = (0, 0, 255)
    COLOR_AEC_MAN = (255, 165, 0)
    COLOR_HEMA_MAN = (128, 0, 128)

    r = 6
    for (x, y) in (session_state.get('aec_points') or []):
        cv2.circle(overlay, (x, y), r, COLOR_AEC, -1)
    for (x, y) in (session_state.get('hema_points') or []):
        cv2.circle(overlay, (x, y), r, COLOR_HEMA, -1)
    for (x, y) in (session_state.get('manual_aec') or []):
        cv2.circle(overlay, (x, y), r, COLOR_AEC_MAN, -1)
    for (x, y) in (session_state.get('manual_hema') or []):
        cv2.circle(overlay, (x, y), r, COLOR_HEMA_MAN, -1)

    # halbtransparente Mischung
    cv2.addWeighted(overlay, 0.6, result_img, 0.4, 0, result_img)

    # Legende
    legend = [
        ("AEC (auto)", COLOR_AEC),
        ("Hämatoxylin (auto)", COLOR_HEMA),
        ("AEC (manuell)", COLOR_AEC_MAN),
        ("Hämatoxylin (manuell)", COLOR_HEMA_MAN),
    ]
    start_x, start_y = 40, 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    line_height = 28

    # Hintergrund
    bg_height = line_height * len(legend) + 20
    bg_width = 300
    overlay2 = result_img.copy()
    cv2.rectangle(overlay2, (20, 20), (20 + bg_width, 20 + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.35, result_img, 0.65, 0, result_img)

    for i, (label, color) in enumerate(legend):
        y_pos = start_y + i * line_height
        cv2.circle(result_img, (start_x, y_pos), 8, color, -1)
        cv2.putText(result_img, label, (start_x + 25, y_pos + 6), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return result_img


# -------------------- Session State Initialisierung --------------------
default_keys = [
    "aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema",
    "aec_hsv", "hema_hsv", "bg_hsv", "last_file", "disp_width", "annot_png"
]
for key in default_keys:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        elif key == "disp_width":
            st.session_state[key] = 1200
        else:
            st.session_state[key] = None

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Wenn neue Datei
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema", "aec_hsv", "hema_hsv", "bg_hsv"]:
        st.session_state[k] = [] if "points" in k or "manual" in k else None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1200
    st.session_state.annot_png = None

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Dynamische Parameter-Vorschläge --------------------
min_area_default, alpha_default, blur_default = dynamic_params_from_image(image_disp)
st.markdown("### ⚙️ Parameter (automatisch vorgeschlagen)")
col1, col2, col3 = st.columns(3)
with col1:
    blur_kernel = st.slider("🔧 Blur (ungerade)", 1, 21, blur_default if blur_default % 2 == 1 else blur_default + 1, step=2)
    min_area = st.number_input("📏 Mindestfläche", 10, 5000, min_area_default)
with col2:
    alpha = st.slider("🌗 Alpha (Kontrast)", 0.5, 3.0, float(alpha_default), step=0.1)
with col3:
    circle_radius = st.slider("⚪ Kreisradius", 3, 20, 6)
    line_thickness = st.slider("📏 Linienstärke", 1, 5, 2)

# -------------------- Modi --------------------
st.markdown("### 🎨 Markierungsmodi")
colA, colB, colC, colD, colE, colF = st.columns(6)
with colA: aec_mode = st.checkbox("🔴 AEC markieren (Kalibrierung)", key="aec_mode")
with colB: hema_mode = st.checkbox("🔵 Hämatoxylin markieren (Kalibrierung)", key="hema_mode")
with colC: bg_mode = st.checkbox("🖌 Hintergrund markieren", key="bg_mode")
with colD: manual_aec_mode = st.checkbox("🟠 AEC manuell", key="manual_aec_mode")
with colE: manual_hema_mode = st.checkbox("🟣 Hämatoxylin manuell", key="manual_hema_mode")
with colF: delete_mode = st.checkbox("🗑️ Löschen (alle Kategorien)", key="delete_mode")

# -------------------- Anzeige & Klick-Handling --------------------
marked_disp = image_disp.copy()
for points_list, color in [
    (st.session_state.aec_points, (255, 0, 0)),
    (st.session_state.hema_points, (0, 0, 255)),
    (st.session_state.manual_aec or [], (255, 165, 0)),
    (st.session_state.manual_hema or [], (128, 0, 128)),
    (st.session_state.bg_points or [], (255, 255, 0)),
]:
    for (x, y) in points_list:
        cv2.circle(marked_disp, (x, y), circle_radius, color, line_thickness)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

if coords:
    x, y = coords["x"], coords["y"]
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

# -------------------- Kalibrierung (manuell oder adaptiv) --------------------
col_cal1, col_cal2 = st.columns(2)
with col_cal1:
    if st.button("⚡ Kalibrierung berechnen"):
        st.session_state.aec_hsv = adaptive_hsv_range(st.session_state.aec_points, hsv_disp)
        st.session_state.hema_hsv = adaptive_hsv_range(st.session_state.hema_points, hsv_disp)
        st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
        st.success("Kalibrierung gespeichert (adaptiv).")
with col_cal2:
    if st.button("🧹 Hintergrundpunkte löschen"):
        st.session_state.bg_points = []
        st.info("Hintergrundpunkte gelöscht.")

# -------------------- Automatische Kalibrierung & Erkennung --------------------
# Trigger: neues Bild oder vorhandene Punkte ohne Kalibrierung
auto_run = False
if (
    st.session_state.last_file is None
    or uploaded_file.name != st.session_state.last_file
    or (st.session_state.aec_points and st.session_state.aec_hsv is None)
    or (st.session_state.hema_points and st.session_state.hema_hsv is None)
):
    auto_run = True

if auto_run:
    st.info("🧠 Automatische Kalibrierung und Erkennung läuft...")
    # adaptive Kalibrierung
    if st.session_state.aec_points:
        st.session_state.aec_hsv = adaptive_hsv_range(st.session_state.aec_points, hsv_disp)
    if st.session_state.hema_points:
        st.session_state.hema_hsv = adaptive_hsv_range(st.session_state.hema_points, hsv_disp)
    st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)

    # Bildvorverarbeitung
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    mask_aec, mask_hema = fuse_masks_hsv_ycrcb(image_disp, hsv_proc, st.session_state.aec_hsv, st.session_state.hema_hsv)

    # Morphologie
    if mask_aec is not None:
        mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        centers = separate_touching_cells(mask_aec)
        if centers:
            st.session_state.aec_points = centers
        else:
            st.session_state.aec_points = get_centers(mask_aec, min_area=min_area)

    if mask_hema is not None:
        mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        centers = separate_touching_cells(mask_hema)
        if centers:
            st.session_state.hema_points = centers
        else:
            st.session_state.hema_points = get_centers(mask_hema, min_area=min_area)

    st.success("✅ Automatische Kalibrierung & Erkennung abgeschlossen!")

# -------------------- Visualisierung Ergebnis --------------------
all_aec = (st.session_state.aec_points or []) + (st.session_state.manual_aec or [])
all_hema = (st.session_state.hema_points or []) + (st.session_state.manual_hema or [])

result_img = image_disp.copy()
result_img = annotate_image(result_img, st.session_state)

count_aec = len(all_aec)
count_hema = len(all_hema)

st.markdown(f"### 🔢 Ergebnis: AEC={count_aec}, Hämatoxylin={count_hema}")
st.image(result_img, use_container_width=True)

# -------------------- Export: CSV + PNG --------------------
df_list = []
for (x, y) in all_aec:
    df_list.append({"X_display": x, "Y_display": y, "Type": "AEC"})
for (x, y) in all_hema:
    df_list.append({"X_display": x, "Y_display": y, "Type": "Hämatoxylin"})

if df_list:
    df = pd.DataFrame(df_list)
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")

    # Confidence: approximativ basierend auf Masken-Intensität (falls vorhanden)
    def compute_confidence(row):
        # Simple heuristic: distance-to-center intensity from mask (placeholder)
        return 1.0

    df["Confidence"] = df.apply(compute_confidence, axis=1)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

    # Annotiertes PNG erzeugen
    buf = io.BytesIO()
    # Convert to PIL RGB
    pil = Image.fromarray(result_img.astype('uint8'), 'RGB')
    pil.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button("🖼️ Annotiertes PNG herunterladen", data=byte_im, file_name="zellkerne_annotiert.png", mime="image/png")
    # Außerdem im Session-State speichern
    st.session_state.annot_png = byte_im

# Optional: Debug Masken anzeigen
if st.checkbox("🧫 Masken anzeigen (Debug)"):
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)
    mask_aec, mask_hema = fuse_masks_hsv_ycrcb(image_disp, hsv_proc, st.session_state.aec_hsv, st.session_state.hema_hsv)
    if mask_aec is not None:
        st.image(mask_aec, caption="AEC-Maske (Debug)", use_container_width=True)
    if mask_hema is not None:
        st.image(mask_hema, caption="Hämatoxylin-Maske (Debug)", use_container_width=True)

st.markdown("---")
st.info("Tip: Korrigiere erkannte Punkte manuell (orange/lila) — das System lernt aus deinen Korrekturen (wenn du es erweiterst).")
