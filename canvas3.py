# canvas_safe_v3.py (Fortsetzung)
# Klicklogik fortführen
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

# -------------------- Kalibrierung --------------------
col_cal1, col_cal2 = st.columns(2)
with col_cal1:
    if st.button("⚡ Kalibrierung berechnen"):
        st.session_state.aec_hsv = adaptive_hsv_range(st.session_state.aec_points, hsv_disp)
        st.session_state.hema_hsv = adaptive_hsv_range(st.session_state.hema_points, hsv_disp)
        st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
        st.success("Kalibrierung gespeichert.")
with col_cal2:
    if st.button("🧹 Hintergrundpunkte löschen"):
        st.session_state.bg_points = []
        st.info("Hintergrundpunkte gelöscht.")

# -------------------- Auto-Erkennung --------------------
if st.button("🤖 Auto-Erkennung"):
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (blur_kernel, blur_kernel), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    mask_aec, mask_hema = fuse_masks_hsv_ycrcb(proc, hsv_proc, st.session_state.aec_hsv, st.session_state.hema_hsv)
    if mask_aec is not None:
        st.session_state.aec_points = separate_touching_cells(mask_aec)
    if mask_hema is not None:
        st.session_state.hema_points = separate_touching_cells(mask_hema)

# -------------------- Ergebnisanzeige --------------------
all_aec = (st.session_state.aec_points or []) + (st.session_state.manual_aec or [])
all_hema = (st.session_state.hema_points or []) + (st.session_state.manual_hema or [])

result_img = annotate_image(image_disp.copy(), st.session_state)
result_img_uint8 = np.clip(result_img, 0, 255).astype(np.uint8)

st.markdown(f"### 🔢 Gesamt: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)}")
st.image(result_img_uint8, use_container_width=True)

# -------------------- CSV & PNG Export --------------------
df_list = []
for (x, y) in all_aec:
    df_list.append({"X_display": x, "Y_display": y, "Type": "AEC"})
for (x, y) in all_hema:
    df_list.append({"X_display": x, "Y_display": y, "Type": "Hämatoxylin"})

if df_list:
    df = pd.DataFrame(df_list)
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    df["Confidence"] = 1.0  # Platzhalter für spätere Scores

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

    # Annotiertes PNG speichern
    buf = io.BytesIO()
    pil_img = Image.fromarray(result_img_uint8, 'RGB')
    pil_img.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button("🖼️ Annotiertes PNG herunterladen", data=byte_im, file_name="zellkerne_annotiert.png", mime="image/png")
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
