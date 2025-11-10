# canvas2_auto_calib_final.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
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
    res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00",0):
                cx = int(round(M["m10"]/M["m00"]))
                cy = int(round(M["m01"]/M["m00"]))
                centers.append((cx, cy))
    return centers

def compute_hsv_range(points, hsv_img, radius=5):
    if not points:
        return None
    vals = []
    for (x,y) in points:
        x_min = max(0, x-radius)
        x_max = min(hsv_img.shape[1], x+radius+1)
        y_min = max(0, y-radius)
        y_max = min(hsv_img.shape[0], y+radius+1)
        region = hsv_img[y_min:y_max, x_min:x_max]
        if region.size>0:
            vals.append(region.reshape(-1,3))
    if not vals:
        return None
    vals = np.vstack(vals)
    h,s,v = vals[:,0].astype(int), vals[:,1].astype(int), vals[:,2].astype(int)
    h_med, s_med, v_med = float(np.median(h)), float(np.median(s)), float(np.median(v))
    n_points = len(points)
    tol_h = int(min(25, 10+n_points*3))
    tol_s = int(min(80, 30+n_points*10))
    tol_v = int(min(80, 30+n_points*10))
    if np.mean(h)>150 or np.mean(h)<20:
        h_med = float(np.median(np.where(h<90, h+180,h)) %180)
        tol_h = min(40, tol_h+5)
    h_min = int(round((h_med-tol_h)%180))
    h_max = int(round((h_med+tol_h)%180))
    s_min = max(0, int(round(s_med-tol_s)))
    s_max = min(255, int(round(s_med+tol_s)))
    v_min = max(0, int(round(v_med-tol_v)))
    v_max = min(255, int(round(v_med+tol_v)))
    return (h_min,h_max,s_min,s_max,v_min,v_max)

def apply_hue_wrap(hsv_img, hmin,hmax,smin,smax,vmin,vmax):
    if hmin<=hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0,smin,vmin]), np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin,smin,vmin]), np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

def ensure_odd(k):
    return k if k%2==1 else k+1

def save_last_calibration(path="kalibrierung.json"):
    data = {k: (v.tolist() if isinstance(v,np.ndarray) else v) for k,v in st.session_state.items() if k in ["aec_hsv","hema_hsv","bg_hsv"]}
    try:
        with open(path,"w") as f:
            json.dump(data,f)
        st.success("💾 Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")

def load_last_calibration(path="kalibrierung.json"):
    try:
        with open(path,"r") as f:
            data = json.load(f)
        for k in ["aec_hsv","hema_hsv","bg_hsv"]:
            st.session_state[k] = np.array(data.get(k)) if data.get(k) else None
        st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (Auto-Kalib)", layout="wide")
st.title("🧬 Zellkern-Zähler – Auto-Kalibrierung")

# -------------------- Session State: sichere Initialisierung --------------------
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
        elif key == "last_auto_run":
            st.session_state[key] = 0
        else:
            st.session_state[key] = []

# Defensive Sicherung: falls last_auto_run mal None ist
if st.session_state.last_auto_run is None:
    st.session_state.last_auto_run = 0


# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points","hema_cal_points","bg_cal_points","aec_auto","hema_auto","manual_aec","manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.slider("📐 Bildbreite",400,2000,st.session_state.disp_width,step=100)
st.session_state.disp_width = DISPLAY_WIDTH
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H_orig*scale)),interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur",1,21,5))
min_area = st.sidebar.number_input("📏 Mindestfläche",10,2000,100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)",0.1,3.0,1.0,step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)",1,20,5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius",1,15,5)
min_points_calib = st.sidebar.slider("🧮 Minimale Punkte für Auto-Kalib",1,10,3)

st.sidebar.markdown("### 🎨 Modus")
mode = st.sidebar.radio("Modus",["Keine","AEC Kalibrier-Punkt setzen","Hämatoxylin Kalibrier-Punkt setzen",
                                   "Hintergrund Kalibrier-Punkt setzen","AEC manuell hinzufügen",
                                   "Hämatoxylin manuell hinzufügen","Punkt löschen"],index=0)

# -------------------- Clean Start beim Moduswechsel --------------------
if mode=="AEC Kalibrier-Punkt setzen": st.session_state.aec_cal_points=[]
elif mode=="Hämatoxylin Kalibrier-Punkt setzen": st.session_state.hema_cal_points=[]
elif mode=="Hintergrund Kalibrier-Punkt setzen": st.session_state.bg_cal_points=[]
elif mode=="AEC manuell hinzufügen": st.session_state.manual_aec=[]
elif mode=="Hämatoxylin manuell hinzufügen": st.session_state.manual_hema=[]

aec_mode = mode=="AEC Kalibrier-Punkt setzen"
hema_mode = mode=="Hämatoxylin Kalibrier-Punkt setzen"
bg_mode = mode=="Hintergrund Kalibrier-Punkt setzen"
manual_aec_mode = mode=="AEC manuell hinzufügen"
manual_hema_mode = mode=="Hämatoxylin manuell hinzufügen"
delete_mode = mode=="Punkt löschen"

st.sidebar.markdown("### ⚡ Schnellaktionen")
if st.sidebar.button("🧹 Alles löschen"):
    for k in keys:
        if "points" in k or "_auto" in k or "manual" in k: st.session_state[k]=[]
        else: st.session_state[k]=None

if st.sidebar.button("💾 Letzte Kalibrierung speichern"): save_last_calibration()
if st.sidebar.button("📂 Letzte Kalibrierung laden"): load_last_calibration()

# -------------------- Bildanzeige + Klicklogik --------------------
marked_disp = image_disp.copy()
for (x,y) in st.session_state.aec_cal_points: cv2.circle(marked_disp,(x,y),max(2,circle_radius//2),(0,120,200),-1)
for (x,y) in st.session_state.hema_cal_points: cv2.circle(marked_disp,(x,y),max(2,circle_radius//2),(200,120,0),-1)
for (x,y) in st.session_state.bg_cal_points: cv2.circle(marked_disp,(x,y),max(2,circle_radius//2),(200,200,0),-1)
for (x,y) in st.session_state.manual_aec: cv2.circle(marked_disp,(x,y),circle_radius,(0,165,255),-1)
for (x,y) in st.session_state.manual_hema: cv2.circle(marked_disp,(x,y),circle_radius,(128,0,128),-1)
for (x,y) in st.session_state.aec_auto: cv2.circle(marked_disp,(x,y),circle_radius,(0,0,255),2)
for (x,y) in st.session_state.hema_auto: cv2.circle(marked_disp,(x,y),circle_radius,(255,0,0),2)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key="clickable_image", width=DISPLAY_WIDTH)

if coords:
    x,y=int(coords["x"]),int(coords["y"])
    if delete_mode:
        for key in ["manual_aec","manual_hema","aec_cal_points","hema_cal_points","bg_cal_points"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),circle_radius)]
    elif aec_mode: st.session_state.aec_cal_points.append((x,y))
    elif hema_mode: st.session_state.hema_cal_points.append((x,y))
    elif bg_mode: st.session_state.bg_cal_points.append((x,y))
    elif manual_aec_mode: st.session_state.manual_aec.append((x,y))
    elif manual_hema_mode: st.session_state.manual_hema.append((x,y))

for k in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k],min_dist=max(4,circle_radius//2))

# -------------------- Auto-Kalibrierung --------------------
def auto_cal(category, cal_key, hsv_key):
    pts = st.session_state.get(cal_key,[])
    if len(pts)>=min_points_calib:
        hsv = compute_hsv_range(pts,hsv_disp,radius=calib_radius)
        if hsv is not None:
            st.session_state[hsv_key]=hsv
            st.success(f"✅ {category}: Auto-Kalibrierung ({len(pts)} Punkte)")
            st.session_state[cal_key]=[]
            st.session_state.last_auto_run += 1

auto_cal("AEC","aec_cal_points","aec_hsv")
auto_cal("Hämatoxylin","hema_cal_points","hema_hsv")
if len(st.session_state.bg_cal_points)>=min_points_calib:
    hsv_bg = compute_hsv_range(st.session_state.bg_cal_points,hsv_disp,radius=calib_radius)
    if hsv_bg is not None:
        st.session_state.bg_hsv = hsv_bg
        st.session_state.bg_cal_points=[]
        st.session_state.last_auto_run += 1

# -------------------- Auto-Erkennung --------------------
if st.session_state.last_auto_run>0:
    proc = cv2.convertScaleAbs(image_disp,alpha=alpha,beta=0)
    if blur_kernel>1: proc = cv2.GaussianBlur(proc,(ensure_odd(blur_kernel),ensure_odd(blur_kernel)),0)
    hsv_proc = cv2.cvtColor(proc,cv2.COLOR_RGB2HSV)

    mask_aec = apply_hue_wrap(hsv_proc,*st.session_state.aec_hsv) if st.session_state.aec_hsv is not None else np.zeros(hsv_proc.shape[:2],dtype=np.uint8)
    mask_hema = apply_hue_wrap(hsv_proc,*st.session_state.hema_hsv) if st.session_state.hema_hsv is not None else np.zeros(hsv_proc.shape[:2],dtype=np.uint8)
    if st.session_state.bg_hsv is not None:
        mask_bg = apply_hue_wrap(hsv_proc,*st.session_state.bg_hsv)
        mask_aec = cv2.bitwise_and(mask_aec,cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema,cv2.bitwise_not(mask_bg))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_aec = cv2.morphologyEx(mask_aec,cv2.MORPH_OPEN,kernel)
    mask_hema = cv2.morphologyEx(mask_hema,cv2.MORPH_OPEN,kernel)

    st.session_state.aec_auto = dedup_points(get_centers(mask_aec,min_area=int(min_area)),min_dist=max(4,circle_radius//2))
    st.session_state.hema_auto = dedup_points(get_centers(mask_hema,min_area=int(min_area)),min_dist=max(4,circle_radius//2))
    st.session_state.last_auto_run = 0

# -------------------- Anzeige + CSV --------------------
aec_auto, aec_manual = st.session_state.aec_auto, st.session_state.manual_aec
hema_auto, hema_manual = st.session_state.hema_auto, st.session_state.manual_hema

st.markdown("### 📊 Ergebnisse")
colA,colB = st.columns(2)
with colA:
    st.metric("AEC (auto)",len(aec_auto))
    st.metric("AEC (manuell)",len(aec_manual))
with colB:
    st.metric("Hämatoxylin (auto)",len(hema_auto))
    st.metric("Hämatoxylin (manuell)",len(hema_manual))

result_img = image_disp.copy()
for x,y in aec_auto: cv2.circle(result_img,(x,y),circle_radius,(0,0,255),2)
for x,y in hema_auto: cv2.circle(result_img,(x,y),circle_radius,(255,0,0),2)
for x,y in aec_manual: cv2.circle(result_img,(x,y),circle_radius,(0,165,255),-1)
for x,y in hema_manual: cv2.circle(result_img,(x,y),circle_radius,(128,0,128),-1)
st.image(result_img, caption="Erkannte Punkte (auto=outline, manuell=filled)", use_column_width=True)

rows=[]
for x,y in aec_auto: rows.append({"X_display":x,"Y_display":y,"Type":"AEC","Source":"auto"})
for x,y in aec_manual: rows.append({"X_display":x,"Y_display":y,"Type":"AEC","Source":"manual"})
for x,y in hema_auto: rows.append({"X_display":x,"Y_display":y,"Type":"Hämatoxylin","Source":"auto"})
for x,y in hema_manual: rows.append({"X_display":x,"Y_display":y,"Type":"Hämatoxylin","Source":"manual"})

if rows:
    df = pd.DataFrame(rows)
    df["X_original"] = (df["X_display"]/scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"]/scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren", data=df.to_csv(index=False).encode("utf-8"), file_name="zellkerne_final.csv", mime="text/csv")

# -------------------- Debug Info --------------------
with st.expander("🧠 Debug Info"):
    st.write({k:st.session_state.get(k) for k in keys})
