import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            M = cv2.moments(c)
            if M.get("m00",0):
                cx = int(round(M["m10"]/M["m00"]))
                cy = int(round(M["m01"]/M["m00"]))
                centers.append((cx,cy))
    return centers

def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

def ensure_odd(k):
    return k if k %2 == 1 else k+1

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler – Schnellmanuell", layout="wide")
st.title("🧬 Zellkern-Zähler – Schneller manueller Modus")

# -------------------- Session-State --------------------
default_keys = [
    "aec_hsv","hema_hsv","bg_hsv",
    "aec_auto","hema_auto",
    "manual_aec","manual_hema",
    "temp_manual_aec","temp_manual_hema",
    "last_file","disp_width"
]
for k in default_keys:
    if k not in st.session_state:
        st.session_state[k] = [] if "auto" in k or "manual" in k or "temp" in k else None

# -------------------- Bild laden --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.last_file = uploaded_file.name
    st.session_state.aec_auto = []
    st.session_state.hema_auto = []
    st.session_state.manual_aec = []
    st.session_state.manual_hema = []
    st.session_state.temp_manual_aec = []
    st.session_state.temp_manual_hema = []

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### 🎨 Modus auswählen")
mode = st.sidebar.radio("Modus",
                        ["Manuell AEC","Manuell Hämatoxylin","Punkt löschen"])
circle_radius = st.sidebar.slider("Kreisradius (Display-Px)",1,20,5)

# -------------------- Bildanzeige + Klick --------------------
from streamlit_image_coordinates import streamlit_image_coordinates

marked_disp = image_disp.copy()
# Auto-Punkte
for x,y in st.session_state.aec_auto:
    cv2.circle(marked_disp,(x,y),circle_radius,(0,0,255),2)
for x,y in st.session_state.hema_auto:
    cv2.circle(marked_disp,(x,y),circle_radius,(255,0,0),2)
# Endgültige manuelle Punkte
for x,y in st.session_state.manual_aec:
    cv2.circle(marked_disp,(x,y),circle_radius,(0,165,255),-1)
for x,y in st.session_state.manual_hema:
    cv2.circle(marked_disp,(x,y),circle_radius,(128,0,128),-1)
# Temporäre Punkte
for x,y in st.session_state.temp_manual_aec:
    cv2.circle(marked_disp,(x,y),circle_radius,(0,255,255),-1)
for x,y in st.session_state.temp_manual_hema:
    cv2.circle(marked_disp,(x,y),circle_radius,(255,0,255),-1)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords:
    x,y = int(coords["x"]),int(coords["y"])
    if mode=="Manuell AEC":
        st.session_state.temp_manual_aec.append((x,y))
    elif mode=="Manuell Hämatoxylin":
        st.session_state.temp_manual_hema.append((x,y))
    elif mode=="Punkt löschen":
        st.session_state.manual_aec = [p for p in st.session_state.manual_aec if not is_near(p,(x,y),circle_radius)]
        st.session_state.manual_hema = [p for p in st.session_state.manual_hema if not is_near(p,(x,y),circle_radius)]

# -------------------- Fertig / Abbrechen Buttons --------------------
col1,col2 = st.columns(2)
with col1:
    if st.button("✅ Fertig (Punkte übernehmen)"):
        st.session_state.manual_aec += st.session_state.temp_manual_aec
        st.session_state.manual_hema += st.session_state.temp_manual_hema
        st.session_state.temp_manual_aec = []
        st.session_state.temp_manual_hema = []
        st.success("Temporäre Punkte übernommen.")
with col2:
    if st.button("❌ Abbrechen (temporäre Punkte verwerfen)"):
        st.session_state.temp_manual_aec = []
        st.session_state.temp_manual_hema = []
        st.info("Temporäre Punkte verworfen.")

# -------------------- Ergebnisse / CSV --------------------
st.markdown("### 📊 Ergebnisse")
st.metric("AEC (manuell)",len(st.session_state.manual_aec))
st.metric("Hämatoxylin (manuell)",len(st.session_state.manual_hema))

rows=[]
for x,y in st.session_state.manual_aec:
    rows.append({"X":x,"Y":y,"Type":"AEC"})
for x,y in st.session_state.manual_hema:
    rows.append({"X":x,"Y":y,"Type":"Hämatoxylin"})
if rows:
    df = pd.DataFrame(rows)
    df["X_orig"] = (df["X"]/scale).round().astype("Int64")
    df["Y_orig"] = (df["Y"]/scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren", df.to_csv(index=False).encode("utf-8"), "manual_points.csv","text/csv")

# -------------------- Debug Info --------------------
with st.expander("🧠 Debug Info"):
    st.write({
        "temp_manual_aec": st.session_state.temp_manual_aec,
        "temp_manual_hema": st.session_state.temp_manual_hema,
        "manual_aec": st.session_state.manual_aec,
        "manual_hema": st.session_state.manual_hema
    })
