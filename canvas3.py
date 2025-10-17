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
    return (h_min,h_max,s_min,s_max,v_min,v_max)

def apply_hue_wrap(hsv_img,hmin,hmax,smin,smax,vmin,vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmin]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

# -------------------- Session State --------------------
def init_session_state():
    keys = ["aec_points","hema_points","bg_points","manual_aec","manual_hema",
            "aec_hsv","hema_hsv","bg_hsv","last_file","disp_width"]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = [] if "points" in key or "manual" in key else None
    if st.session_state.get("disp_width") is None:
        st.session_state.disp_width = 1400

# -------------------- Safe-Display --------------------
def safe_display(img):
    if img is None:
        st.warning("Kein Bild zum Anzeigen")
        return
    try:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img_uint8 = np.clip(img,0,255).astype(np.uint8)
        st.image(img_uint8, use_column_width=True)
    except Exception as e:
        st.error(f"Fehler beim Anzeigen des Bildes: {e}")

# -------------------- Auto-Erkennung --------------------
def auto_detect(image_disp, min_area=50, blur_kernel=5):
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    proc = cv2.convertScaleAbs(image_disp, alpha=1.0, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc,(blur_kernel,blur_kernel),0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    # AEC
    if st.session_state.aec_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.aec_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        st.session_state.aec_points = get_centers(mask,min_area)

    # HEMA
    if st.session_state.hema_hsv:
        hmin,hmax,smin,smax,vmin,vmax = st.session_state.hema_hsv
        mask = apply_hue_wrap(hsv_proc,hmin,hmax,smin,smax,vmin,vmax)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        st.session_state.hema_points = get_centers(mask,min_area)

# -------------------- Punkte auf Bild zeichnen --------------------
def draw_points(image_disp):
    marked_disp = image_disp.copy()
    for points_list,color in [
        (st.session_state.aec_points,(255,0,0)),
        (st.session_state.hema_points,(0,0,255)),
        (st.session_state.manual_aec,(255,165,0)),
        (st.session_state.manual_hema,(128,0,128)),
        (st.session_state.bg_points,(255,255,0))
    ]:
        for (x,y) in points_list:
            cv2.circle(marked_disp,(x,y),6,color,2)
    return marked_disp

# -------------------- Haupt-App --------------------
def main():
    st.set_page_config(page_title="Zellkern-Zähler Modular", layout="wide")
    st.title("🧬 Zellkern-Zähler – Modular & stabil")
    init_session_state()

    # -------------------- Bild Upload --------------------
    uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
    if uploaded_file is None:
        st.info("Bitte zuerst ein Bild hochladen.")
        st.stop()
    image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
    H,W = image_orig.shape[:2]

    # -------------------- Anzeigegröße --------------------
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH
    scale = DISPLAY_WIDTH / W
    image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H*scale)))

    # -------------------- Modus-Auswahl --------------------
    colA,colB,colC = st.columns(3)
    with colA: aec_mode = st.checkbox("🔴 AEC markieren")
    with colB: hema_mode = st.checkbox("🔵 HEMA markieren")
    with colC: bg_mode = st.checkbox("🖌 Hintergrund markieren")

    # -------------------- Klicklogik --------------------
    coords = streamlit_image_coordinates(Image.fromarray(image_disp), key="clickable_image", width=DISPLAY_WIDTH)
    if coords:
        x,y = coords["x"], coords["y"]
        if aec_mode: st.session_state.aec_points.append((x,y))
        elif hema_mode: st.session_state.hema_points.append((x,y))
        elif bg_mode: st.session_state.bg_points.append((x,y))

    # -------------------- Kalibrierung --------------------
    if st.button("⚡ Kalibrierung speichern"):
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, cv2.cvtColor(image_disp,cv2.COLOR_RGB2HSV))
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, cv2.cvtColor(image_disp,cv2.COLOR_RGB2HSV))
        st.success("Kalibrierung gespeichert!")

    # -------------------- Auto-Erkennung --------------------
    if st.button("🤖 Auto-Erkennung"):
        auto_detect(image_disp)

    # -------------------- Punkte zeichnen und finale Anzeige --------------------
    marked_disp = draw_points(image_disp)
    safe_display(marked_disp)

    # -------------------- CSV Export --------------------
    all_points = (st.session_state.aec_points or []) + (st.session_state.hema_points or [])
    if all_points:
        df_list = []
        for (x,y) in st.session_state.aec_points: df_list.append({"X":x,"Y":y,"Type":"AEC"})
        for (x,y) in st.session_state.hema_points: df_list.append({"X":x,"Y":y,"Type":"HEMA"})
        df = pd.DataFrame(df_list)
        df["X_original"] = (df["X"]/scale).round().astype("Int64")
        df["Y_original"] = (df["Y"]/scale).round().astype("Int64")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

if __name__=="__main__":
    main()
