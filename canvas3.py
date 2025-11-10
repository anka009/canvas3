# canvas3.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from pathlib import Path
import json

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
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
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

# -------------------- Kalibrierung speichern/laden --------------------
def save_last_calibration(path="kalibrierung.json"):
    data = {
        "aec_hsv": st.session_state.aec_hsv.tolist() if st.session_state.aec_hsv is not None else None,
        "hema_hsv": st.session_state.hema_hsv.tolist() if st.session_state.hema_hsv is not None else None,
        "bg_hsv": st.session_state.bg_hsv.tolist() if st.session_state.bg_hsv is not None else None
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

# -------------------- Session State --------------------
default_keys = [
    "aec_cal_points","hema_cal_points","bg_cal_points",
    "aec_auto","hema_auto","manual_aec","manual_hema",
    "aec_hsv","hema_hsv","bg_hsv",
    "last_file","disp_width","last_auto_run"
]
for key in default_keys:
    if key not in st.session_state:
        if key in ["aec_hsv","hema_hsv","bg_hsv"]:
            st.session_state[key] = None
        elif key=="disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = []

# -------------------- Bild hochladen --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset state bei neuem Bild
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points","hema_cal_points","bg_cal_points","aec_auto","hema_auto","manual_aec","manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.sidebar.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Parameter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, 1))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 2000, 100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, 0.1)
circle_radius = st.sidebar.slider("⚪ Punktradius", 1, 20, 6)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius", 1, 15, 5)
min_points_calib = st.sidebar.slider("🧮 Min. Punkte für Auto-Kalib.",1,10,3)
st.sidebar.info("Kalibrierung startet automatisch, sobald Punktzahl erreicht ist.")

mode = st.sidebar.radio("Modus auswählen",[
    "Keine",
    "AEC Kalibrier-Punkt setzen",
    "Hämatoxylin Kalibrier-Punkt setzen",
    "Hintergrund Kalibrier-Punkt setzen",
    "AEC manuell hinzufügen",
    "Hämatoxylin manuell hinzufügen",
    "Punkt löschen"
])

aec_mode = mode=="AEC Kalibrier-Punkt setzen"
hema_mode = mode=="Hämatoxylin Kalibrier-Punkt setzen"
bg_mode = mode=="Hintergrund Kalibrier-Punkt setzen"
manual_aec_mode = mode=="AEC manuell hinzufügen"
manual_hema_mode = mode=="Hämatoxylin manuell hinzufügen"
delete_mode = mode=="Punkt löschen"

if st.sidebar.button("🧹 Alles löschen"):
    for k in ["aec_cal_points","hema_cal_points","bg_cal_points","aec_auto","hema_auto","manual_aec","manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = None
    st.success("Alles gelöscht.")

if st.sidebar.button("💾 Letzte Kalibrierung speichern"):
    save_last_calibration()
if st.sidebar.button("📂 Letzte Kalibrierung laden"):
    load_last_calibration()

# -------------------- Canvas --------------------
disp_img = image_disp.copy()
canvas_result = st_canvas(
    fill_color="rgba(255,0,0,0.3)",
    stroke_width=circle_radius,
    stroke_color="#000000",
    background_image=Image.fromarray(disp_img),
    update_streamlit=True,
    height=disp_img.shape[0],
    width=disp_img.shape[1],
    drawing_mode="point",
    key="canvas_click"
)

# -------------------- Klicklogik --------------------
if canvas_result.json_data is not None:
    for obj in canvas_result.json_data["objects"]:
        if "left" in obj and "top" in obj:
            x = int(round(obj["left"]))
            y = int(round(obj["top"]))

            if delete_mode:
                for key in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema","aec_auto","hema_auto"]:
                    st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),circle_radius)]
            elif aec_mode:
                st.session_state.aec_cal_points.append((x,y))
            elif hema_mode:
                st.session_state.hema_cal_points.append((x,y))
            elif bg_mode:
                st.session_state.bg_cal_points.append((x,y))
            elif manual_aec_mode:
                st.session_state.manual_aec.append((x,y))
            elif manual_hema_mode:
                st.session_state.manual_hema.append((x,y))

# Dedup
for k in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k],min_dist=max(4,circle_radius//2))

# -------------------- Auto-Kalibrierung --------------------
def auto_calib(name, cal_key, hsv_key):
    pts = st.session_state.get(cal_key,[])
    if len(pts) >= min_points_calib:
        hsv = compute_hsv_range(pts,hsv_disp,radius=calib_radius)
        if hsv:
            st.session_state[hsv_key] = hsv
            st.success(f"✅ {name}: Kalibrierung automatisch ({len(pts)} Punkte)")
            st.session_state[cal_key] = []

auto_calib("AEC","aec_cal_points","aec_hsv")
auto_calib("Hämatoxylin","hema_cal_points","hema_hsv")
auto_calib("Hintergrund","bg_cal_points","bg_hsv")

# -------------------- Auto-Erkennung --------------------
if st.session_state.aec_hsv or st.session_state.hema_hsv:
    proc = cv2.convertScaleAbs(image_disp,alpha=alpha)
    if blur_kernel>1:
        proc = cv2.GaussianBlur(proc,(ensure_odd(blur_kernel),ensure_odd(blur_kernel)),0)
    hsv_proc = cv2.cvtColor(proc,cv2.COLOR_RGB2HSV)

    mask_aec = apply_hue_wrap(hsv_proc,*map(int,st.session_state.aec_hsv)) if st.session_state.aec_hsv else np.zeros(hsv_proc.shape[:2],dtype=np.uint8)
    mask_hema = apply_hue_wrap(hsv_proc,*map(int,st.session_state.hema_hsv)) if st.session_state.hema_hsv else np.zeros(hsv_proc.shape[:2],dtype=np.uint8)
    if st.session_state.bg_hsv:
        mask_bg = apply_hue_wrap(hsv_proc,*map(int,st.session_state.bg_hsv))
        mask_aec = cv2.bitwise_and(mask_aec,cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema,cv2.bitwise_not(mask_bg))

    st.session_state.aec_auto = dedup_points(get_centers(mask_aec,int(min_area)),min_dist=max(4,circle_radius//2))
    st.session_state.hema_auto = dedup_points(get_centers(mask_hema,int(min_area)),min_dist=max(4,circle_radius//2))

# -------------------- Anzeige & CSV --------------------
def draw_points(img, points, color, thickness=-1, outline=False):
    for (x,y) in points:
        if outline:
            cv2.circle(img,(x,y),circle_radius,color,2)
        else:
            cv2.circle(img,(x,y),circle_radius,color,thickness)

result_img = image_disp.copy()
draw_points(result_img, st.session_state.aec_auto,(0,0,255),outline=True)
draw_points(result_img, st.session_state.hema_auto,(255,0,0),outline=True)
draw_points(result_img, st.session_state.manual_aec,(0,165,255))
draw_points(result_img, st.session_state.manual_hema,(128,0,128))
st.image(result_img, caption="Erkannte Punkte (auto=outline, manuell=filled)")

# CSV
rows=[]
for x,y in st.session_state.aec_auto:
    rows.append({"X":x,"Y":y,"Type":"AEC","Source":"auto"})
for x,y in st.session_state.manual_aec:
    rows.append({"X":x,"Y":y,"Type":"AEC","Source":"manual"})
for x,y in st.session_state.hema_auto:
    rows.append({"X":x,"Y":y,"Type":"Hämatoxylin","Source":"auto"})
for x,y in st.session_state.manual_hema:
    rows.append({"X":x,"Y":y,"Type":"Hämatoxylin","Source":"manual"})

if rows:
    df = pd.DataFrame(rows)
    df["X_orig"] = (df["X"]/scale).round().astype("Int64")
    df["Y_orig"] = (df["Y"]/scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren",data=df.to_csv(index=False).encode("utf-8"),file_name="zellkerne.csv",mime="text/csv")
