# 🧠 Detección de Tumores Cerebrales por MRI – Aplicación Streamlit

Clasificador **binario (sí/no)** para detección de tumores cerebrales en imágenes **MRI** con explicación visual usando **Grad-CAM**.  
Aplicación web construida con **Streamlit** y modelo **PyTorch (ResNet18)** preentrenado.

---

## 📦 Estructura del Repositorio

```
.
├── .streamlit/           # Carpeta de configuración de Streamlit
├── LICENSE              # Licencia Apache 2.0
├── Procfile            # Definición de proceso para despliegue (Render)
├── README.md           # Este archivo
├── best_threshold.json # Umbrales sugeridos (youden_J, max_F1, recall_priority)
├── requirements.txt    # Dependencias para despliegue/entorno
├── runtime.txt         # Versión de Python para despliegue (3.10 recomendado)
└── streamlit_app.py    # Aplicación principal Streamlit (subir imagen → predicción → Grad-CAM)
```

---

## 🧠 ¿Qué hace?

1. **Subes** una imagen (PNG/JPG/TIFF) de una resonancia magnética.
2. La aplicación la **preprocesa** (escala de grises→3 canales, redimensión 224, normalización ImageNet).
3. El modelo **infiere** `prob_yes` (probabilidad de presencia de tumor).
4. Se compara contra un **umbral** (por defecto: *recall_priority* de `best_threshold.json`).
5. Si la predicción es **SÍ**, genera una visualización **Grad-CAM** + **caja delimitadora** para resaltar la región de mayor atención.

---

## 🔧 Requisitos

- Python **3.10** (recomendado)
- Compatible con CPU (funciona sin GPU)
- Paquetes listados en `requirements.txt`

> **Nota**: El archivo del modelo preentrenado (`resnet18_best.pt`) debe agregarse al repositorio. Si el archivo es grande, considera usar **Git LFS**:
>
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> git commit -m "chore(lfs): habilitar LFS para archivos .pt"
> ```

---

## ▶️ Ejecutar Localmente

```bash
# 1) Crear y activar entorno virtual
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 3) Ejecutar la aplicación
streamlit run streamlit_app.py
# Se abre en: http://localhost:8501
```

---

## ⚙️ Umbrales de Decisión

`best_threshold.json` contiene tres opciones de umbral:

- **youden_J**: Balance entre especificidad/sensibilidad en curva ROC
- **max_F1**: Maximiza el puntaje F1 en curva PR  
- **recall_priority** (por defecto): Prioriza el recall para reducir falsos negativos

Puedes cambiar el umbral activo editando el archivo JSON o ajustando 1-2 líneas en `streamlit_app.py`.

---

## 🖼️ Uso

1. Sube una imagen de MRI
2. Haz clic en **"🔍 Predecir"**
3. Visualiza los resultados:
   - **Predicción**: SÍ (anomalía detectada) o NO
   - **prob_yes** (puntuación de probabilidad 0-1)
   - Si es SÍ: Visualización Grad-CAM con caja delimitadora estimada

---

## 🚢 Despliegue en Render (Tier Gratuito)

1. Sube el repositorio a GitHub
2. Crea un Web Service en Render → Python
3. Asegúrate de tener:
   - `requirements.txt`
   - `runtime.txt` con `3.10`
   - `Procfile` con:
     ```
     web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
     ```
4. Render instalará las dependencias e iniciará la aplicación
5. Accede en: `https://<tu-servicio>.onrender.com`

> **Nota del Tier Gratuito**: La aplicación entra en suspensión sin tráfico; la primera solicitud puede tomar varios segundos en despertar.

---

## ❓ Preguntas Frecuentes

**¿Puedo usar imágenes diferentes al conjunto de datos de entrenamiento?**  
Sí, siempre que sean imágenes MRI en escala de grises o RGB. La aplicación convierte a 3 canales y normaliza apropiadamente.

**¿Por qué mi prob_yes es exactamente 1.000 o 0.000?**  
El modelo podría estar muy confiado. Considera ajustar el umbral, calibrar probabilidades o revisar el conjunto de validación.

**¿La caja delimitadora del Grad-CAM es una localización clínica real?**  
Es explicativa, no detección clínica. Para localización robusta, considera entrenar un detector/segmentador con anotaciones (ej., YOLO/Faster-RCNN/U-Net).

---

## 🧪 Notas Técnicas

- **Arquitectura**: ResNet18 (capa final reentrenada para 2 clases)
- **Preprocesamiento**: Escala de grises→3 canales, Redimensión(224), Normalización(ImageNet)
- **Explicabilidad**: Grad-CAM en la última capa conv de layer4
- **Métricas/Umbrales**: Pre-generadas (ROC/PR) y guardadas en `best_threshold.json`

---
## 📄 Licencia
Este proyecto se distribuye bajo la licencia Apache-2.0. Consulta el archivo [LICENSE](./LICENSE) para más detalles.

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
