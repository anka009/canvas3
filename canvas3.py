# canvas_smart.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, segmentation
import io

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler — Smart", layout="wide")
st.title("🧬 Zellkern-Zähler — Smart Version")

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1)-np.array(p2))<r

def get_centers(mask, min_area=50):
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00",0) != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

def compute_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=25):
    if not points: return None
    vals = np.array([hsv_img[y,x] for (x,y) in points])
    h = vals[:,0].astype(int)
    s = vals[:,1].astype(int)
    v = vals[:,2].astype(int)
    h_min = max(0,np.min(h)-buffer_h)
    h_max = min(180,np.max(h)+buffer_h)
    s_min = max(0,np.min(s)-buffer_s)
    s_max = min(255,np.max(s)+buffer_s)
    v_min = max(0,np.min(v)-buffer_v)
    v_max = min(255,np.max(v)+buffer_v)
    return (h_min,h_max,s_min,s_max,v_min,v_max)

def apply_hue_wrap(hsv_img,hmin,hmax,smin,smax,vmin,vmax):
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img,np.array([hmin,smin,vmin]),np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img,np.array([0,smin,vmin]),np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img,np.array([hmin,smin,vmin]),np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo,mask_hi)
    return mask

def annotate_image(img, session_state):
    overlay = img.copy()
    COLORS = {'aec_points':(255,0,0),'hema_points':(0,0,255),
              'manual_aec':(255,165,0),'manual_hema':(128,0,128)}
    for key,color in COLORS.items():
        for (x,y) in session_state.get(key,[]):
            cv2.circle(overlay,(x,y),6,color,-1)
    cv2.addWeighted(overlay.astype(np.uint8),0.6,img.astype(np.uint8),0.4,0,img)
    return img

# -------------------- Session State --------------------
default_keys = ["aec_points","hema_points","bg_points","manual_aec","manual_hema",
                "aec_hsv","hema_hsv","bg_hsv","last_file","disp_width","annot_png"]
for key in default_keys:
    if key not in st.session_state:
        st.session_state[key] = [] if "points" in key or "manual" in key else None
if st.session_state.get('disp_width') is None:
    st.session_state.disp_width = 1200

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema",
              "aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = [] if "points" in k or "manual" in k else None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400

DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH/W_orig
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H_orig*scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp,cv2.COLOR_RGB2HSV)

# -------------------- Modi --------------------
colA,colB,colC,colD,colE,colF = st.columns(6)
with colA: aec_mode = st.checkbox("🔴 AEC markieren")
with colB: hema_mode = st.checkbox("🔵 Hämatoxylin markieren")
with colC: bg_mode = st.checkbox("🖌 Hintergrund markieren")
with colD: manual_aec_mode = st.checkbox("🟠 AEC manuell")
with colE: manual_hema_mode = st.checkbox("🟣 Hämatoxylin manuell")
with colF: delete_mode = st.checkbox("🗑️ Löschen")

# -------------------- Bildanzeige & Klicks --------------------
marked_disp = image_disp.copy()
for points_list,color in [(st.session_state.aec_points,(255,0,0)),
                          (st.session_state.hema_points,(0,0,255)),
                          (st.session_state.manual_aec or [],(255,165,0)),
                          (st.session_state.manual_hema or [],(128,0,128)),
                          (st.session_state.bg_points or [],(255,255,0))]:
    for (x,y) in points_list:
        cv2.circle(marked_disp,(x,y),6,color,2)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp),
                                     key="clickable_image", width=DISPLAY_WIDTH)
if coords:
    x,y = coords["x"],coords["y"]
    if delete_mode:
        for key in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),6)]
    elif aec_mode: st.session_state.aec_points.append((x,y))
    elif hema_mode: st.session_state.hema_points.append((x,y))
    elif bg_mode: st.session_state.bg_points.append((x,y))
    elif manual_aec_mode: st.session_state.manual_aec.append((x,y))
    elif manual_hema_mode: st.session_state.manual_hema.append((x,y))

# -------------------- Kalibrierung --------------------
col_cal1,col_cal2 = st.columns(2)
with col_cal1:
    if st.button("⚡ Kalibrierung berechnen"):
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points,hsv_disp)
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points,hsv_disp)
        st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points,hsv_disp)
        st.success("Kalibrierung gespeichert.")
with col_cal2:
    if st.button("🧹 Hintergrundpunkte löschen"):
        st.session_state.bg_points=[]
        st.info("Hintergrundpunkte gelöscht.")

# -------------------- Smart Auto-Erkennung --------------------
if st.button("🤖 Smart Auto-Erkennung"):
    proc = cv2.convertScaleAbs(image_disp, alpha=1.0, beta=0)
    hsv_proc = cv2.cvtColor(proc,cv2.COLOR_RGB2HSV)
    aec_hsv = compute_hsv_range(st.session_state.aec_points,hsv_proc) if st.session_state.aec_points else None
    hema_hsv = compute_hsv_range(st.session_state.hema_points,hsv_proc) if st.session_state.hema_points else None

    mask_aec = apply_hue_wrap(hsv_proc, *(aec_hsv or (0,0,0,0,0,0)))
    mask_hema = apply_hue_wrap(hsv_proc, *(hema_hsv or (0,0,0,0,0,0)))

    # Touching-Cell Separation
    centers_aec=[]
    if mask_aec is not None and cv2.countNonZero(mask_aec)>0:
        distance = cv2.distanceTransform(mask_aec, cv2.DIST_L2, 5)
        local_max = (distance==ndi.maximum_filter(distance,size=15))
        markers = measure.label(local_max)
        labels = segmentation.watershed(-distance, markers, mask=mask_aec)
        for region in measure.regionprops(labels):
            if region.area>=20:
                cy,cx = region.centroid
                centers_aec.append((int(cx),int(cy)))
    centers_hema = get_centers(mask_hema,min_area=50)

    st.session_state.aec_points = centers_aec
    st.session_state.hema_points = centers_hema
    st.success(f"Smart-Erkennung abgeschlossen: AEC={len(centers_aec)}, HEMA={len(centers_hema)}")

# -------------------- Sicheres Anzeigen & Export --------------------
all_aec = (st.session_state.aec_points or []) + (st.session_state.manual_aec or [])
all_hema = (st.session_state.hema_points or []) + (st.session_state.manual_hema or [])
result_img = annotate_image(image_disp.copy(), st.session_state)

if result_img is not None and isinstance(result_img,np.ndarray):
    if len(result_img.shape)==2:
        result_img_rgb = cv2.cvtColor(result_img,cv2.COLOR_GRAY2RGB)
    elif result_img.shape[-1]==4:
        result_img_rgb = result_img[:,:,:3]
    elif result_img.shape[-1]==3:
        result_img_rgb = result_img
    else:
        result_img_rgb = cv2.cvtColor(result_img,cv2.COLOR_GRAY2RGB)

    result_img_uint8 = np.clip(result_img_rgb,0,255).astype(np.uint8)
    st.markdown(f"### 🔢 Gesamt: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)}")
    st.image(result_img_uint8,use_container_width=True)

    # CSV Export
    df_list=[]
    for (x,y) in all_aec: df_list.append({"X_display":x,"Y_display":y,"Type":"AEC"})
    for (x,y) in all_hema: df_list.append({"X_display":x,"Y_display":y,"Type":"Hämatoxylin"})
    if df_list:
        df=pd.DataFrame(df_list)
        df["X_original"]=(df["X_display"]/scale).round().astype("Int64")
        df["Y_original"]=(df["Y_display"]/scale).round().astype("Int64")
        df["Confidence"]=1.0

        csv=df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV exportieren",data=csv,file_name="zellkerne.csv",mime="text/csv")

        # Annotiertes PNG
        buf=io.BytesIO()
        pil_img=Image.fromarray(result_img_uint8,'RGB')
        pil_img.save(buf,format='PNG')
        byte_im=buf.getvalue()
        st.download_button("🖼️ Annotiertes PNG herunterladen",data=byte_im,file_name="zellkerne_annotiert.png",mime="image/png")
        st.session_state.annot_png
