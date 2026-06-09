import streamlit as st
from datetime import datetime
import re
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Sistema ANPR - Placas Colombianas",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Modo claro (por defecto) ── */
    :root {
        --primary-color:  #2563eb;
        --success-color:  #10b981;
        --warning-color:  #f59e0b;
        --error-color:    #ef4444;
        --bg-color:       #f8fafc;
        --card-bg:        #ffffff;
        --text-primary:   #1e293b;
        --text-secondary: #64748b;
        --border-color:   #e2e8f0;

        --alert-success-bg: #d1fae5;
        --alert-warning-bg: #fef3c7;
        --alert-error-bg:   #fee2e2;
        --alert-info-bg:    #dbeafe;
        --debug-bg:         #f1f5f9;
        --debug-border:     #cbd5e1;
    }

    /* ── Modo oscuro automático ── */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color:  #60a5fa;
            --success-color:  #34d399;
            --warning-color:  #fbbf24;
            --error-color:    #f87171;
            --bg-color:       #0f172a;
            --card-bg:        #1e293b;
            --text-primary:   #f1f5f9;
            --text-secondary: #94a3b8;
            --border-color:   #334155;

            --alert-success-bg: #064e3b;
            --alert-warning-bg: #451a03;
            --alert-error-bg:   #450a0a;
            --alert-info-bg:    #1e3a5f;
            --debug-bg:         #1e293b;
            --debug-border:     #334155;
        }
    }

    .stApp {
        background-color: var(--bg-color);
    }

    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
    }

    .stMetric {
        background-color: var(--card-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .alert-success {
        background-color: var(--alert-success-bg);
        border-left: 4px solid var(--success-color);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }

    .alert-warning {
        background-color: var(--alert-warning-bg);
        border-left: 4px solid var(--warning-color);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }

    .alert-error {
        background-color: var(--alert-error-bg);
        border-left: 4px solid var(--error-color);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }

    .alert-info {
        background-color: var(--alert-info-bg);
        border-left: 4px solid var(--primary-color);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
    }

    .debug-box {
        background-color: var(--debug-bg);
        border: 1px solid var(--debug-border);
        color: var(--text-primary);
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

if 'detecciones_historial' not in st.session_state:
    st.session_state.detecciones_historial = []

if 'contador_detecciones' not in st.session_state:
    st.session_state.contador_detecciones = 0

def mostrar_alerta(tipo, mensaje):
    alert_class = f"alert-{tipo}"
    st.markdown(f'<div class="{alert_class}">{mensaje}</div>', unsafe_allow_html=True)


def mostrar_debug(titulo, contenido):
    st.markdown(f'<div class="debug-box"><strong>{titulo}:</strong> {contenido}</div>', unsafe_allow_html=True)

@st.cache_resource
def cargar_modelos():
    with st.spinner("Inicializando modelos..."):
        try:
            reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            st.error(f"Error al cargar EasyOCR: {e}")
            return None, None

        modelo_path = None
        rutas_posibles = ["runs/best.pt", "modelo_final/best.pt"]

        for ruta in rutas_posibles:
            if Path(ruta).exists():
                modelo_path = ruta
                break

        if not modelo_path:
            runs_dir = Path("runs/detect")
            experimentos = list(runs_dir.glob("*/weights/best.pt"))
            if experimentos:
                modelo_path = str(max(experimentos, key=lambda p: p.stat().st_mtime))

        if modelo_path:
            try:
                detector = YOLO(modelo_path)
                st.sidebar.success(f"Modelo cargado: {Path(modelo_path).name}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
                detector = YOLO("yolov8s.pt")
                st.sidebar.warning("Usando YOLOv8s genérico")
        else:
            detector = YOLO("yolov8s.pt")
            st.sidebar.warning("Modelo no encontrado. Usando YOLOv8s")

        return reader, detector


reader, detector = cargar_modelos()

def clasificar_tipo_placa(placa_crop, texto):
    texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())

    hsv = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2HSV)

    rangos_color = {
        'amarillo': ([15, 80, 80], [40, 255, 255]),
        'blanco': ([0, 0, 170], [180, 50, 255]),
        'azul': ([90, 50, 50], [140, 255, 255]),
        'verde': ([35, 40, 40], [90, 255, 255]),
        'rojo': ([0, 70, 50], [10, 255, 255])
    }

    porcentajes = {}
    for color, (bajo, alto) in rangos_color.items():
        mask = cv2.inRange(hsv, np.array(bajo), np.array(alto))
        porcentajes[color] = np.sum(mask > 0) / mask.size

    es_moto = bool(re.match(r"^[A-Z]{3}\d{2}[A-Z]$", texto_limpio))

    if es_moto:
        if porcentajes['amarillo'] > 0.15:
            return "Motocicleta Particular"
        elif porcentajes['blanco'] > 0.15:
            return "Motocicleta Pública"
        return "Motocicleta"

    if porcentajes['azul'] > 0.20:
        prefijos = {"CD": "Cuerpo Diplomático", "CC": "Cuerpo Consular",
                    "OI": "Organismo Internacional", "AT": "Personal Administrativo"}
        for prefijo, tipo in prefijos.items():
            if texto_limpio.startswith(prefijo):
                return tipo
        return "Vehículo Diplomático"

    if porcentajes['verde'] > 0.20:
        if texto_limpio.startswith("R"):
            return "Remolque"
        elif texto_limpio.startswith("S"):
            return "Semirremolque"
        return "Vehículo Oficial"

    if porcentajes['rojo'] > 0.20:
        return "Vehículo de Carga Pública"

    if porcentajes['blanco'] > 0.25:
        return "Servicio Público"

    if porcentajes['amarillo'] > 0.15:
        return "Vehículo Particular"

    if porcentajes['azul'] > 0.08 and porcentajes['blanco'] > 0.08:
        return "Vehículo Clásico"

    return "Tipo No Identificado"


def analizar_placa(texto_ocr):
    placa = re.sub(r'[^A-Z0-9]', '', texto_ocr.strip().upper())

    if len(placa) == 6:
        letras = placa[:3].replace("0", "O").replace("1", "I").replace("2", "Z").replace("8", "B")
        numeros = placa[3:].replace("O", "0").replace("I", "1").replace("Z", "2").replace("B", "8")
        placa = letras + numeros
    elif len(placa) == 7:
        letras = placa[:3].replace("0", "O").replace("1", "I")
        numeros = placa[3:6].replace("O", "0").replace("I", "1")
        final = placa[6].replace("0", "O").replace("1", "I")
        placa = letras + numeros + final

    formatos_validos = {
        'carro_tradicional': r"^[A-Z]{3}\d{3}$",
        'carro_mercosur': r"^[A-Z]{3}\d{3}[A-Z]$",
        'moto': r"^[A-Z]{3}\d{2}[A-Z]$"
    }

    tipo_formato = None
    for tipo, patron in formatos_validos.items():
        if re.match(patron, placa):
            tipo_formato = tipo
            break

    if not tipo_formato:
        return None, f"Formato inválido: {placa}"

    if tipo_formato == 'moto':
        digito = int(placa[-2])
        formato = "Motocicleta"
    elif tipo_formato == 'carro_mercosur':
        digito = int(placa[-2])
        formato = "Mercosur"
    else:
        digito = int(placa[-1])
        formato = "Tradicional"

    dia_semana = datetime.now().weekday()
    hora_actual = datetime.now().hour
    en_horario = 6 <= hora_actual < 20

    tabla_restriccion = {
        0: [4, 5, 6, 7],
        1: [8, 9, 0, 1],
        2: [2, 3, 4, 5],
        3: [6, 7, 8, 9],
        4: [0, 1, 2, 3]
    }

    if dia_semana >= 5:
        estado = "Libre - Fin de semana"
        restriccion_activa = False
    elif not en_horario:
        estado = "Libre - Fuera de horario"
        restriccion_activa = False
    elif digito in tabla_restriccion[dia_semana]:
        estado = "RESTRICCIÓN ACTIVA"
        restriccion_activa = True
    else:
        estado = "Libre circulación"
        restriccion_activa = False

    return {
        "placa": placa,
        "digito": digito,
        "estado": estado,
        "formato": formato,
        "restriccion_activa": restriccion_activa
    }, None

def registrar_deteccion(placa, tipo_vehiculo, formato, restriccion_activa, estado):
    deteccion = {
        "timestamp": datetime.now(),
        "hora": datetime.now().hour,
        "placa": placa,
        "tipo_vehiculo": tipo_vehiculo,
        "formato": formato,
        "estado": estado,
        "restriccion_activa": restriccion_activa,
        "es_moto": "Moto" in tipo_vehiculo or "Motocicleta" in tipo_vehiculo
    }

    st.session_state.detecciones_historial.append(deteccion)
    st.session_state.contador_detecciones += 1

def mostrar_dashboard():
    st.header("Panel de Análisis y Estadísticas")

    if not st.session_state.detecciones_historial:
        st.info("No hay datos disponibles. Procese imágenes en la pestaña de Detección.")
        return

    df = pd.DataFrame(st.session_state.detecciones_historial)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Detecciones", len(df))

    with col2:
        motos = df[df['es_moto'] == True]
        st.metric("Motocicletas", len(motos))

    with col3:
        con_restriccion = df[df['restriccion_activa'] == True]
        st.metric("Con Restricción", len(con_restriccion))

    with col4:
        sin_restriccion = df[df['restriccion_activa'] == False]
        st.metric("Sin Restricción", len(sin_restriccion))

    st.divider()

    st.subheader("Distribución por Tipo de Vehículo")

    conteo_tipos = df['tipo_vehiculo'].value_counts().reset_index()
    conteo_tipos.columns = ['Tipo de Vehículo', 'Cantidad']

    fig_barras = px.bar(
        conteo_tipos,
        x='Tipo de Vehículo',
        y='Cantidad',
        color='Tipo de Vehículo',
        title='Cantidad de Placas Detectadas por Tipo',
        text='Cantidad',
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig_barras.update_layout(xaxis_tickangle=-45, showlegend=False, height=400)
    st.plotly_chart(fig_barras, use_container_width=True)

    tipo_mas_comun = conteo_tipos.iloc[0]
    porcentaje_motos = len(motos) / len(df) * 100

    mostrar_alerta('info', f"""
        <strong>Análisis:</strong> El tipo más detectado es <strong>{tipo_mas_comun['Tipo de Vehículo']}</strong> 
        con {tipo_mas_comun['Cantidad']} detecciones. Las motocicletas representan el {porcentaje_motos:.1f}% del total.
    """)

    st.divider()

    st.subheader("Estado de Restricción")

    restriccion_data = df['restriccion_activa'].value_counts().reset_index()
    restriccion_data.columns = ['Estado', 'Cantidad']
    restriccion_data['Estado'] = restriccion_data['Estado'].map({
        True: 'Con Restricción',
        False: 'Sin Restricción'
    })

    fig_pie = px.pie(
        restriccion_data,
        values='Cantidad',
        names='Estado',
        title='Distribución de Restricciones',
        color='Estado',
        color_discrete_map={
            'Con Restricción': '#ef4444',
            'Sin Restricción': '#10b981'
        }
    )

    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

    porcentaje_restringidos = len(con_restriccion) / len(df) * 100

    mostrar_alerta('info', f"""
        <strong>Análisis:</strong> El {porcentaje_restringidos:.1f}% presenta restricción activa. 
        El {100 - porcentaje_restringidos:.1f}% tiene libre circulación.
    """)

    st.divider()

    st.subheader("Detecciones por Hora del Día")

    detecciones_por_hora = df.groupby('hora').size().reset_index(name='cantidad')
    todas_horas = pd.DataFrame({'hora': range(24)})
    detecciones_por_hora = todas_horas.merge(detecciones_por_hora, on='hora', how='left').fillna(0)

    fig_linea = go.Figure()

    fig_linea.add_trace(go.Scatter(
        x=detecciones_por_hora['hora'],
        y=detecciones_por_hora['cantidad'],
        mode='lines+markers',
        name='Detecciones',
        line=dict(color='#2563eb', width=3),
        marker=dict(size=10)
    ))

    fig_linea.add_vrect(x0=6, x1=9, fillcolor="green", opacity=0.1, layer="below", line_width=0,
                        annotation_text="Hora Pico AM", annotation_position="top left")
    fig_linea.add_vrect(x0=17, x1=20, fillcolor="red", opacity=0.1, layer="below", line_width=0,
                        annotation_text="Hora Pico PM", annotation_position="top left")

    fig_linea.update_layout(
        title='Patrón de Detecciones por Hora',
        xaxis_title='Hora del Día',
        yaxis_title='Número de Detecciones',
        height=400,
        xaxis=dict(dtick=1)
    )

    st.plotly_chart(fig_linea, use_container_width=True)

    hora_max = detecciones_por_hora.loc[detecciones_por_hora['cantidad'].idxmax()]

    mostrar_alerta('info', f"""
        <strong>Análisis:</strong> La hora con mayor actividad es las {int(hora_max['hora'])}:00 con {int(hora_max['cantidad'])} detecciones.
    """)

    st.divider()

    st.subheader("Historial de Detecciones")

    df_display = df[['timestamp', 'placa', 'tipo_vehiculo', 'formato', 'estado']].copy()
    df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df_display.columns = ['Fecha y Hora', 'Placa', 'Tipo de Vehículo', 'Formato', 'Estado']

    st.dataframe(df_display, use_container_width=True, height=300)

st.title("Sistema de Detección y Reconocimiento de Placas")
st.caption("Implementación con YOLOv8, EasyOCR y Análisis de Restricciones Vehiculares")

with st.sidebar:
    st.header("Configuración del Sistema")

    confianza = st.slider(
        "Umbral de confianza",
        min_value=0.01,
        max_value=0.90,
        value=0.10,
        step=0.01,
        help="Ajuste la sensibilidad. Valores bajos (0.01-0.20) para placas difíciles."
    )

    mostrar_debug_info = st.checkbox(
        "Mostrar información de diagnóstico",
        value=True,
        help="Muestra detalles del proceso de detección en cada etapa."
    )

    mostrar_preprocesamiento = st.checkbox(
        "Mostrar preprocesamiento OCR",
        value=False,
        help="Visualiza las transformaciones aplicadas a la imagen."
    )

    st.divider()

    st.header("Gestión de Datos")

    if st.button("Limpiar historial"):
        st.session_state.detecciones_historial = []
        st.session_state.contador_detecciones = 0
        st.success("Historial limpiado")
        st.rerun()

    st.metric("Detecciones registradas", st.session_state.contador_detecciones)

    st.divider()

    st.header("Información del Sistema")
    st.markdown("""
    **Tecnologías:**
    - YOLOv8: Detección de objetos
    - EasyOCR: Reconocimiento óptico
    - OpenCV: Procesamiento de imágenes
    - Plotly: Visualización de datos
    """)

tab_detector, tab_dashboard = st.tabs(["Detección de Placas", "Panel de Análisis"])

with tab_detector:
    st.header("Procesamiento de Imágenes")

    archivo = st.file_uploader(
        "Seleccione una imagen para procesar",
        type=["jpg", "jpeg", "png"],
        help="Formatos: JPG, JPEG, PNG. Resolución mínima recomendada: 1280x720."
    )

    if archivo:

        file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_original = img.copy()

        if mostrar_debug_info:
            mostrar_debug("Imagen cargada", f"Tamaño: {img.shape[1]}x{img.shape[0]} píxeles")

        col_imagen, col_resultados = st.columns([1, 1])

        with st.spinner("Procesando imagen..."):
            resultados = detector(img, conf=confianza, imgsz=1280)

        detecciones = []

        with col_imagen:
            st.subheader("Imagen Procesada")

        with col_resultados:
            st.subheader("Diagnóstico del Proceso")

        num_boxes = sum(len(r.boxes) for r in resultados)

        if mostrar_debug_info:
            with col_resultados:
                mostrar_debug("Detecciones YOLO",
                              f"Se encontraron {num_boxes} posibles placas con confianza >= {confianza:.2f}")

        for r in resultados:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_det = float(box.conf[0])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                placa_crop = img_original[y1:y2, x1:x2]

                if placa_crop.size == 0:
                    if mostrar_debug_info:
                        with col_resultados:
                            mostrar_debug("Error", "Recorte vacío")
                    continue

                if mostrar_debug_info:
                    with col_resultados:
                        mostrar_debug("Recorte de placa",
                                      f"Tamaño: {placa_crop.shape[1]}x{placa_crop.shape[0]} píxeles | Confianza YOLO: {conf_det:.2%}")

                gray = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                imagenes_ocr = [
                    gray,
                    cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                    cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                    cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
                ]

                if mostrar_preprocesamiento:
                    with col_imagen:
                        st.image(imagenes_ocr[1], caption="Preprocesamiento OCR", width=300)

                mejor_texto = ""
                mejor_conf = 0
                todos_textos = []

                for img_ocr in imagenes_ocr:
                    resultados_ocr = reader.readtext(img_ocr, detail=1)

                    for (_, texto, conf_ocr) in resultados_ocr:
                        texto_limpio = re.sub(r'[^A-Z0-9]', '', texto.upper())
                        todos_textos.append(f"'{texto_limpio}' ({conf_ocr:.2%})")

                        if len(texto_limpio) >= 5 and conf_ocr > mejor_conf:
                            mejor_conf = conf_ocr
                            mejor_texto = texto_limpio

                if mostrar_debug_info:
                    with col_resultados:
                        mostrar_debug("Textos OCR encontrados",
                                      f"{len(todos_textos)} textos | Mejor: '{mejor_texto}' ({mejor_conf:.2%})")
                        if todos_textos:
                            st.write("**Todos los textos:**", ", ".join(todos_textos[:10]))

                if mejor_texto:
                    analisis, error_msg = analizar_placa(mejor_texto)

                    if analisis:
                        tipo_vehiculo = clasificar_tipo_placa(placa_crop, mejor_texto)

                        deteccion = {
                            "placa": analisis["placa"],
                            "tipo": tipo_vehiculo,
                            "digito": analisis["digito"],
                            "estado": analisis["estado"],
                            "formato": analisis["formato"],
                            "conf_det": conf_det,
                            "conf_ocr": mejor_conf,
                            "restriccion_activa": analisis["restriccion_activa"]
                        }

                        detecciones.append(deteccion)

                        registrar_deteccion(
                            analisis["placa"],
                            tipo_vehiculo,
                            analisis["formato"],
                            analisis["restriccion_activa"],
                            analisis["estado"]
                        )

                        if mostrar_debug_info:
                            with col_resultados:
                                mostrar_debug("Placa válida", f"{analisis['placa']} - {tipo_vehiculo}")
                    else:
                        if mostrar_debug_info:
                            with col_resultados:
                                mostrar_debug("Placa rechazada", error_msg)
                else:
                    if mostrar_debug_info:
                        with col_resultados:
                            mostrar_debug("Sin texto OCR", "No se pudo leer ningún texto válido")

        with col_imagen:
            st.image(img, channels="BGR", use_container_width=True)

            if detecciones:
                mostrar_alerta('success', f"Se detectaron {len(detecciones)} placa(s) válida(s).")
            else:
                mostrar_alerta('warning', "No se detectaron placas válidas.")

        with col_resultados:
            st.divider()
            st.subheader("Resultados del Análisis")

            if detecciones:
                for i, det in enumerate(detecciones, 1):
                    with st.expander(f"Placa {i}: {det['placa']}", expanded=True):
                        if det['restriccion_activa']:
                            mostrar_alerta('error', f"<strong>Placa:</strong> {det['placa']} - {det['estado']}")
                        else:
                            mostrar_alerta('success', f"<strong>Placa:</strong> {det['placa']} - {det['estado']}")

                        col_tipo, col_formato, col_digito = st.columns(3)
                        col_tipo.metric("Tipo", det["tipo"])
                        col_formato.metric("Formato", det["formato"])
                        col_digito.metric("Dígito", det["digito"])

                        st.caption(f"Confianza detección: {det['conf_det']:.2%} | Confianza OCR: {det['conf_ocr']:.2%}")
            else:
                mostrar_alerta('error', "No se encontraron placas válidas.")

                st.markdown("""
                <strong>Recomendaciones:</strong>
                <ul>
                    <li>Reduzca el umbral de confianza a 0.05-0.15</li>
                    <li>Use imágenes con mayor resolución</li>
                    <li>Mejore la iluminación</li>
                    <li>Evite ángulos muy inclinados</li>
                </ul>
                """, unsafe_allow_html=True)

with tab_dashboard:
    mostrar_dashboard()

st.divider()

st.caption("""
Sistema de Reconocimiento Automático de Placas (ANPR) | 
YOLOv8 + EasyOCR + Visión por Computadora | 
Normativa RUNT Colombia
""")