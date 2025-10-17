# -------------------- Imports --------------------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1)-np.array(p2)) < r

def get_centers(mask, min_area=50):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00",0)!=0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

def compute_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=25):
    if not points: return None
    vals = np.array([hsv_img[y,x] for (x,y) in points])
    h,s,v = vals[:,0], vals[:,1], vals[:,2]
    h_min = max(0,int(np.min(h)-buffer_h))
    h_max = min(180,int(np.max(h)+buffer_h))
    s_min = max(0,int(np.min(s)-buffer_s))
    s_max = min(255,int(np.max(s)+buffer_s))
    v_min = max(0,int(np.min(v)-buffer_v))
    v_max = min(255,int(np.max(v)+buffer_v))
    return (h_min,h_max,s_min,s_max,v_min,vmax)

def apply_hue_wrap(hsv_img,hmin,hmax,smin,smax,vmin,vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

# -------------------- Session State Setup --------------------
def init_session_state():
    keys = ["aec_points","hema_points","bg_points","manual_aec","manual_hema",
            "aec_hsv","hema_hsv","bg_hsv","last_file","disp_width"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = [] if "points" in key or "manual" in key else None
    if st.session_state.get("disp_width") is None:
        st.session_state.disp_width = 1400

# -------------------- Bild-Upload --------------------
def upload_image():
    uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
    if uploaded_file is None:
        st.info("Bitte zuerst ein Bild hochladen.")
        st.stop()
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    return image, uploaded_file.name

# -------------------- Punktmarkierung / Canvas --------------------
def mark_points(image_disp):
    coords = streamlit_image_coordinates(Image.fromarray(image_disp), key="clickable_image", width=image_disp.shape[1])
    return coords

# -------------------- Annotate & Safe Display --------------------
def annotate_and_display(image_disp):
    marked_disp = image_disp.copy()
    # Punkte zeichnen
    for points_list,color in [
        (st.session_state.aec_points,(255,0,0)),
        (st.session_state.hema_points,(0,0,255)),
        (st.session_state.manual_aec,(255,165,0)),
        (st.session_state.manual_hema,(128,0,128)),
        (st.session_state.bg_points,(255,255,0))
    ]:
        for (x,y) in points_list:
            cv2.circle(marked_disp,(x,y),6,color,2)

    # Safe-Display
    if marked_disp is not None and isinstance(marked_disp, np.ndarray):
        if len(marked_disp.shape)==2:
            marked_disp = cv2.cvtColor(marked_disp,cv2.COLOR_GRAY2RGB)
        elif marked_disp.shape[-1]==4:
            marked_disp = cv2.cvtColor(marked_disp,cv2.COLOR_RGBA2RGB)
        marked_disp_uint8 = np.clip(marked_disp,0,255).astype(np.uint8)
        st.image(marked_disp_uint8, use_container_width=True)

# -------------------- Auto-Erkennung (Platzhalter) --------------------
def auto_detect(image_disp):
    # Hier später smarte Erkennung einbauen
    st.info("Auto-Erkennung läuft...")
    hsv_proc = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)
    if st.session_state.aec_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.aec_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        st.session_state.aec_points = get_centers(mask,50)
    if st.session_state.hema_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.hema_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        st.session_state.hema_points = get_centers(mask,50)

# -------------------- CSV Export --------------------
def export_results(scale):
    all_aec = (st.session_state.aec_points or []) + (st.session_state.manual_aec or [])
    all_hema = (st.session_state.hema_points or []) + (st.session_state.manual_hema or [])
    df_list = []
    for (x,y) in all_aec: df_list.append({"X_display":x,"Y_display":y,"Type":"AEC"})
    for (x,y) in all_hema: df_list.append({"X_display":x,"Y_display":y,"Type":"Hämatoxylin"})
    if df_list:
        df = pd.DataFrame(df_list)
        df["X_original"] = (df["X_display"]/scale).round().astype("Int64")
        df["Y_original"] = (df["Y_display"]/scale).round().astype("Int64")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

# -------------------- Haupt-App --------------------
def main():
    st.set_page_config(page_title="Zellkern-Zähler Modular", layout="wide")
    st.title("🧬 Zellkern-Zähler – Modular & stabil")
    init_session_state()

    # Bild-Upload
    image_orig, filename = upload_image()
    H,W = image_orig.shape[:2]

    # Slider für Anzeigegröße
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH
    scale = DISPLAY_WIDTH / W
    image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H*scale)))

    # Punktmarkierung
    coords = mark_points(image_disp)
    if coords:
        x,y = coords["x"], coords["y"]
        st.session_state.aec_points.append((x,y))  # Beispiel, kann alle Modi abdecken

    # Annotation & Safe Display
    annotate_and_display(image_disp)

    # Auto-Erkennung Button
    if st.button("🤖 Auto-Erkennung"):
        auto_detect(image_disp)

    # CSV Export
    export_results(scale)

if __name__=="__main__":
    main()
