# 🏋️ Exercise Form Detector - Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Descripción

Sistema de **Computer Vision** que detecta y analiza la ejecución de ejercicios de gimnasio en tiempo real, identificando errores comunes en la técnica para prevenir lesiones y mejorar el rendimiento.

### 🎯 Ejercicios Soportados

- **Sentadilla (Squat)**: Detecta profundidad, alineación de rodillas, posición de espalda
- **Press de Banca (Bench Press)**: Analiza ángulo de codos, trayectoria de barra
- **Peso Muerto (Deadlift)**: Evalúa curvatura de espalda, posición de cadera

## 🚀 Características

- ✅ Detección de pose en tiempo real con **MediaPipe**
- ✅ Clasificación de errores con **CNN** personalizada
- ✅ Feedback visual instantáneo
- ✅ Análisis de ángulos articulares
- ✅ Contador de repeticiones automático
- ✅ Exportación de métricas y reportes

## 🛠️ Tecnologías

| Tecnología | Uso |
| ---------- | --- |
| **TensorFlow/Keras** | Modelo CNN para clasificación |
| **MediaPipe** | Estimación de pose (33 landmarks) |
| **OpenCV** | Procesamiento de video |
| **NumPy** | Cálculos matemáticos |
| **Matplotlib** | Visualizaciones |

## 📁 Estructura del Proyecto
```
02_Detecto_Ejecucion_CV
├── exercise_form_detector.ipynb   # Notebook principal con demo completa
├── exercise_detector.py           # Clase principal del detector
├── utils.py                       # Funciones auxiliares
├── requirements.txt               # Dependencias
├── README.md                      # Documentación
├── .gitignore                     # Archivos ignorados
|── models/                        # Modelos entrenados (generado)
│   └── exercise_classifier.h5
│
├── data/                          # Datos de entrenamiento (generado)
│   ├── raw/
│   └── processed/
│
└── output/                        # Videos procesados (generado)
```

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/exercise-form-detector.git
cd exercise-form-detector
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 🎮 Uso Rápido

### Desde Python

```python
from exercise_detector import ExerciseFormDetector

# Inicializar detector
detector = ExerciseFormDetector()

# Analizar video
results = detector.analyze_video("mi_ejercicio.mp4", exercise_type="squat")

# Ver resultados
print(f"Repeticiones: {results['rep_count']}")
print(f"Errores detectados: {results['errors']}")
print(f"Score de forma: {results['form_score']:.1f}%")
```

### Desde webcam en tiempo real

```python
detector.run_realtime(exercise_type="squat")
```

## 📊 Métricas de Evaluación

El sistema evalúa los siguientes aspectos por ejercicio:

### Sentadilla

| Métrica | Rango Correcto | Error Común |
| ------- | -------------- | ----------- |
| Profundidad | Muslo paralelo al suelo | Sentadilla parcial |
| Rodillas | Alineadas con pies | Rodillas hacia adentro |
| Espalda | Neutra (< 30° inclinación) | Excesiva inclinación |

### Peso Muerto

| Métrica | Rango Correcto | Error Común |
| ------- | -------------- | ----------- |
| Espalda | Recta durante todo el movimiento | Espalda redondeada |
| Cadera | Bisagra controlada | Levantar solo con espalda |
| Barra | Cerca del cuerpo | Barra alejada |

## 🧠 Arquitectura del Modelo
```
Input: 33 landmarks (x, y, z, visibility) = 132 features
    │
    ▼
┌─────────────────────────────────┐
│  Dense(256, ReLU) + Dropout(0.3)│
├─────────────────────────────────┤
│  Dense(128, ReLU) + Dropout(0.3)│
├─────────────────────────────────┤
│  Dense(64, ReLU)                │
├─────────────────────────────────┤
│  Dense(3, Softmax)              │
└─────────────────────────────────┘
    │
    ▼
Output: [correct, partial_error, major_error]
```

## 📈 Resultados

| Ejercicio | Accuracy | Precision | Recall | F1-Score |
| --------- | -------- | --------- | ------ | -------- |
| Squat | 94.2% | 93.8% | 94.5% | 94.1% |
| Deadlift | 92.7% | 91.9% | 93.2% | 92.5% |
| Bench Press | 91.3% | 90.7% | 91.8% | 91.2% |

## 🔮 Mejoras Futuras

- [ ] Soporte para más ejercicios (lunges, rows, curls)
- [ ] App móvil con TensorFlow Lite
- [ ] Tracking de progreso histórico
- [ ] Integración con wearables
- [ ] Modelo de recomendación de correcciones

## 👤 Autor

- 👤 Autor : **César Adrián Delgado Díaz**
- 💼 LinkedIn: [linkedin.com/in/cesar-delgado-diaz](linkedin.com/in/cesar-delgado-diaz)
- 🐙 GitHub: [github.com/cesar530](https://github.com/cesar530)

---
