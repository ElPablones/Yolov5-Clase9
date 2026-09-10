import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import io
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

st.set_page_config(
    page_title="Evaluación YOLOv5",
    page_icon="🔬",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("🔬 Herramienta de Evaluación YOLOv5")
st.markdown("Prueba la precisión, velocidad y comportamiento del modelo en diferentes escenarios.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("⚙️ Parámetros de Inferencia")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU (Solapamiento)", 0.0, 1.0, 0.45, 0.01)
        
        st.divider()
        st.subheader("Modo de Entrada")
        input_mode = st.radio(
            "Selecciona la fuente:", 
            ["📷 Foto (Cámara)", "📁 Subir Archivo", "🎥 Video en Vivo (WebRTC)"]
        )

    # --- LÓGICA DE PROCESAMIENTO ESTÁTICO (FOTOS / ARCHIVOS) ---
    def process_static_image(image_bytes):
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1] # RGB a BGR para OpenCV/YOLO
        img_h, img_w, _ = np_img.shape

        start_time = time.time()
        results = model(np_img, conf=conf_threshold, iou=iou_threshold)
        inference_time = (time.time() - start_time) * 1000 # milisegundos

        result = results[0]
        boxes = result.boxes
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        # Interfaz de resultados
        st.subheader(f"⏱️ Tiempo de inferencia: {inference_time:.1f} ms")
        
        col_img, col_data = st.columns([3, 2])
        with col_img:
            st.image(annotated_rgb, caption="Resultado de la detección", use_container_width=True)

        with col_data:
            if boxes is not None and len(boxes) > 0:
                st.write("**Mapeo Espacial de Objetos**")
                
                spatial_data = []
                crops = []

                for box in boxes:
                    cat = int(box.cls.item())
                    conf = float(box.conf.item())
                    label = model.names[cat]
                    
                    # Coordenadas (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calcular centroide para ubicarlo en el espacio (Izquierda, Centro, Derecha)
                    x_center = (x1 + x2) / 2
                    if x_center < img_w / 3:
                        ubicacion = "Izquierda"
                    elif x_center > (img_w / 3) * 2:
                        ubicacion = "Derecha"
                    else:
                        ubicacion = "Centro"

                    spatial_data.append({
                        "Objeto": label,
                        "Confianza": f"{conf:.2f}",
                        "Ubicación": ubicacion
                    })

                    # Extraer el recorte (Cropping)
                    crop_img = annotated_rgb[y1:y2, x1:x2]
                    crops.append((label, crop_img))

                # Mostrar tabla de datos espaciales
                st.dataframe(pd.DataFrame(spatial_data), use_container_width=True)
                
                # Galería dinámica de recortes
                st.write("**Objetos Extraídos (Cropping)**")
                crop_cols = st.columns(min(len(crops), 4) if len(crops) > 0 else 1)
                for i, (label, crop) in enumerate(crops):
                    if i < 8: # Limitar a 8 recortes para no saturar la UI
                        with crop_cols[i % 4]:
                            st.image(crop, caption=label, use_container_width=True)
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")

    # --- RUTAS DE ENTRADA ---
    if input_mode == "📷 Foto (Cámara)":
        picture = st.camera_input("Capturar imagen para analizar")
        if picture:
            process_static_image(picture.getvalue())

    elif input_mode == "📁 Subir Archivo":
        uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            process_static_image(uploaded_file.getvalue())

    elif input_mode == "🎥 Video en Vivo (WebRTC)":
        st.info("El video en vivo procesa cuadro por cuadro. Ajusta los parámetros en la barra lateral para ver cómo cambia el rendimiento.")
        
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # Medir FPS e Inferencia internamente
            start_t = time.time()
            results = model(img, conf=conf_threshold, iou=iou_threshold)
            inf_time_ms = (time.time() - start_t) * 1000
            
            annotated_img = results[0].plot()
            
            # Dibujar telemetría directamente en el video
            cv2.putText(
                annotated_img, 
                f"Inferencia: {inf_time_ms:.1f} ms", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        webrtc_streamer(
            key="yolo-eval",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

else:
    st.error("No se pudo cargar el modelo.")
    st.stop()
