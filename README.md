# Sistema de Detección y Reconocimiento de Placas Vehiculares Colombianas

Sistema inteligente para la detección automática de placas de vehículos colombianos, con énfasis en motocicletas. Combina visión por computadora (YOLOv8), reconocimiento óptico de caracteres (EasyOCR) y un dashboard analítico con visualización de datos en tiempo real.



Por: Laura Vanessa Figueredo y Daniel Alejandro Reyes León
## Características principales

- Detección automática de placas mediante YOLOv8
- Reconocimiento óptico de caracteres con EasyOCR
- Clasificación del tipo de vehículo según normativa RUNT
- Análisis de color de placa (HSV) para identificar categoría
- Validación de formatos colombianos (motos, carros tradicionales, Mercosur)
- Cálculo automático de restricción de Pico y Placa en Bogotá
- Dashboard con tres visualizaciones analíticas
- Interfaz web profesional con Streamlit

## Tecnologías utilizadas

- **Python 3.11+**
- **YOLOv8** (Ultralytics) - Detección de objetos
- **EasyOCR** - Reconocimiento óptico de caracteres
- **OpenCV** - Procesamiento de imágenes
- **Streamlit** - Aplicación web
- **Plotly** - Visualización de datos
- **Pandas** - Análisis de datos
- **PyTorch** - Framework de deep learning

## Requisitos del sistema

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| Sistema operativo | Windows 10/11 (64-bit) | Windows 11 |
| Procesador | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| Memoria RAM | 8 GB | 16 GB |
| Almacenamiento | 5 GB libres | 10 GB SSD |
| GPU | No requerida | NVIDIA con CUDA (opcional) |

## Instalación paso a paso

### 1. Instalar Python

Descargar Python 3.11 o superior desde [python.org](https://www.python.org/downloads/). Durante la instalación, marcar la opción **"Add Python to PATH"**.

Verificar la instalación en PowerShell:

```powershell
python --version
```

### 2. Instalar Git

Descargar Git desde [git-scm.com](https://git-scm.com/download/win) e instalar con las opciones predeterminadas.

### 3. Clonar el repositorio

```powershell
cd C:\Users\TuUsuario\Documents
git clone https://github.com/la00429/placas_final_4_June.git
cd placas_final_4_June
```

### 4. Crear y activar el entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Al activarse, el prompt mostrará el prefijo `(.venv)`.

### 5. Instalar dependencias

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

El archivo `requirements.txt` incluye:

```
ultralytics>=8.0.0
easyocr>=1.7.0
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0
torch>=2.0.0
torchvision>=0.15.0
plotly>=5.18.0
pandas>=2.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

### 6. Verificar la instalación

```powershell
python -c "import ultralytics, easyocr, streamlit, cv2, plotly, pandas; print('Instalación exitosa')"
```

## Ejecución de la aplicación

Con el entorno virtual activado:

```powershell
streamlit run app_streamlit.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`.

## Qué se debería ver al ejecutar

### Pantalla principal

La aplicación presenta una interfaz con dos pestañas:

1. **Detección de Placas**: para subir imágenes y procesarlas
2. **Panel de Análisis**: dashboard con estadísticas y visualizaciones

### Panel lateral izquierdo

Contiene los siguientes elementos:

- **Umbral de confianza**: slider para ajustar la sensibilidad de detección (0.10 a 0.90)
- **Mostrar preprocesamiento OCR**: checkbox para visualizar las transformaciones aplicadas
- **Mostrar información de diagnóstico**: checkbox para ver detalles del proceso
- **Limpiar historial**: botón para reiniciar las estadísticas
- **Contador de detecciones**: métrica con el total de placas procesadas
- **Información del sistema**: lista de tecnologías utilizadas

## Flujo de procesamiento

Al subir una imagen, el sistema ejecuta las siguientes etapas:

1. **Carga de imagen**: lectura del archivo JPG, JPEG o PNG
2. **Detección con YOLOv8**: localización de la placa mediante bounding boxes
3. **Recorte de región**: extracción del área donde está la placa
4. **Preprocesamiento OCR**: conversión a escala de grises, redimensionamiento 4x, filtrado gaussiano, binarización Otsu y sharpening
5. **Reconocimiento de texto**: múltiples intentos con EasyOCR seleccionando el resultado con mayor confianza
6. **Corrección heurística**: reemplazo de confusiones comunes (O/0, I/1, B/8, Z/2)
7. **Validación de formato**: verificación contra patrones RUNT
8. **Análisis de color**: clasificación por tipo de vehículo usando espacio HSV
9. **Cálculo de Pico y Placa**: determinación de restricción según día y hora
10. **Registro en historial**: almacenamiento para el dashboard

## Resultados esperados

### En la columna izquierda

- Imagen original con un recuadro verde alrededor de la placa detectada
- Alerta de éxito o advertencia según el resultado

### En la columna derecha

- Placa identificada (ejemplo: `OBW59D`)
- Tipo de vehículo (ejemplo: `Motocicleta Particular`)
- Formato (ejemplo: `Motocicleta`)
- Dígito de restricción (ejemplo: `9`)
- Estado de Pico y Placa (ejemplo: `Libre circulación` o `RESTRICCIÓN ACTIVA`)
- Métricas de confianza de detección y OCR

## Dashboard y visualizaciones

La pestaña **Panel de Análisis** contiene tres visualizaciones interactivas:

### Visualización 1: Gráfico de barras

Muestra la distribución de placas detectadas por tipo de vehículo. Permite identificar el tipo más frecuente y el porcentaje de motocicletas sobre el total.

**Análisis generado**: indica el tipo de vehículo más detectado y el porcentaje de motocicletas.

### Visualización 2: Gráfico circular

Muestra el porcentaje de vehículos con restricción activa de Pico y Placa versus los que tienen libre circulación. Utiliza colores semánticos: rojo para restringidos y verde para libres.

**Análisis generado**: indica el porcentaje de vehículos restringidos y el porcentaje con libre circulación.

### Visualización 3: Gráfico de líneas temporal

Muestra la cantidad de detecciones por hora del día (0 a 23 horas), con marcadores visuales para las horas pico de tráfico (6-9 AM y 5-8 PM).

**Análisis generado**: identifica la hora con mayor actividad y establece correlaciones con los patrones de movilidad urbana.

### Tabla de historial

Registro cronológico con las columnas: Fecha y Hora, Placa, Tipo de Vehículo, Formato y Estado.

## Formatos de placas soportados

| Tipo de vehículo | Formato | Ejemplo | Color |
|------------------|---------|---------|-------|
| Motocicleta particular | AAA99A | OBW59D | Amarillo |
| Vehículo particular tradicional | AAA999 | ABC123 | Amarillo |
| Vehículo particular Mercosur | AAA999A | ABC123D | Amarillo |
| Servicio público | AA999A | AB123C | Blanco |
| Cuerpo diplomático | CD999 | CD123 | Azul |
| Cuerpo consular | CC999 | CC456 | Azul |
| Remolque | R99999 | R12345 | Verde |
| Carga pública | - | - | Rojo |

## Restricción de Pico y Placa - Bogotá D.C.

**Horario de aplicación**: lunes a viernes, 6:00 AM a 8:00 PM

| Día | Dígitos restringidos |
|-----|---------------------|
| Lunes | 4, 5, 6, 7 |
| Martes | 8, 9, 0, 1 |
| Miércoles | 2, 3, 4, 5 |
| Jueves | 6, 7, 8, 9 |
| Viernes | 0, 1, 2, 3 |

**Motocicletas**: restricción los sábados (dígitos 0-4 restringidos, 5-9 libres).

**Fines de semana**: libre circulación para vehículos particulares.

## Entrenamiento del modelo

### Estructura del dataset

```
train/images/ y train/labels/
valid/images/ y valid/labels/
test/images/ y test/labels/
```

Cada imagen debe tener su archivo de anotación `.txt` en formato YOLO.

### Ejecutar entrenamiento

```powershell
python main.py
```

### Parámetros configurados

- Épocas: 50
- Tamaño de imagen: 640x640
- Batch size: 16
- Early stopping con paciencia de 20 épocas

### Duración estimada

- CPU: 4-8 horas
- GPU NVIDIA: 30-60 minutos

### Métricas de referencia

- mAP50 > 0.90: Excelente
- mAP50 entre 0.70 y 0.90: Aceptable
- mAP50 < 0.70: Requiere más datos

## Estructura del proyecto

```
placas_final_4_June/
├── app_streamlit.py      # Aplicación web principal
├── main.py               # Script de entrenamiento
├── requirements.txt      # Dependencias
├── data.yaml             # Configuración del dataset
├── .gitignore            # Archivos ignorados por Git
├── README.md             # Documentación
├── train/                # Dataset de entrenamiento
├── valid/                # Dataset de validación
├── test/                 # Dataset de prueba
├── runs/                 # Resultados de entrenamiento
│   └── best.pt           # Modelo entrenado
└── .venv/                # Entorno virtual
```

## Solución de problemas comunes

### Error: ModuleNotFoundError

Instalar la dependencia faltante:

```powershell
pip install [nombre_del_modulo]
```

### El sistema no detecta placas

- Reducir el umbral de confianza a 0.05-0.20
- Usar imágenes con mayor resolución (mínimo 1280x720)
- Mejorar la iluminación de la imagen
- Evitar ángulos de inclinación superiores a 30 grados
- Activar el modo de diagnóstico para identificar la etapa del fallo

### El OCR lee caracteres incorrectos

- Verificar que la placa esté completamente contenida en el bounding box
- Asegurar iluminación uniforme
- Activar "Mostrar preprocesamiento OCR" para depurar
- Usar imágenes más nítidas

### La aplicación es lenta

- Habilitar aceleración GPU si está disponible
- Reducir el tamaño de imagen de inferencia (cambiar `imgsz=1280` a `imgsz=640`)
- Cerrar aplicaciones que consuman recursos del sistema

### Error: CUDA out of memory

Reducir el batch size en `main.py`:

```python
batch=8
```

## Recomendaciones para capturar imágenes

- Resolución mínima de 1280x720 píxeles
- Iluminación uniforme y adecuada
- Ángulo de inclinación inferior a 30 grados
- Minimizar reflejos y sombras sobre la placa
- Mantener la placa completamente visible en el encuadre
- Evitar imágenes con desenfoque o movimiento

## Licencia

Este proyecto se distribuye bajo la Licencia MIT.

## Referencias

- [Documentación YOLOv8](https://docs.ultralytics.com/)
- [Documentación EasyOCR](https://www.jaided.ai/easyocr/)
- [Documentación Streamlit](https://docs.streamlit.io/)
- [Documentación Plotly](https://plotly.com/python/)
- [Normativa RUNT Colombia](https://www.runt.com.co/)
- Ministerio de Transporte de Colombia - Resolución 1720 de 2014
- Secretaría de Movilidad de Bogotá D.C. - Regulación de Pico y Placa

---

**Versión 2.0 - Junio 2026**

Proyecto desarrollado como aplicación académica de inteligencia artificial para la detección y reconocimiento automatizado de placas vehiculares bajo normativa colombiana, con énfasis en motocicletas y análisis estadístico en tiempo real.