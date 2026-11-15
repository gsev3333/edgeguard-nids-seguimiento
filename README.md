# EDGEGUARD — NIDS BASADO EN EDGE AI (1D-CNN + GOOGLE CORAL TPU)

> Proyecto EdgeGuard (ficha GI-INNOVATIC). Seguimiento del 16 de octubre de 2026.  
> Todo el contenido de este repositorio quedó consolidado en este README para que el profesor lo revise directamente al abrir el proyecto, sin entrar a carpetas adicionales.

---

## ÍNDICE

1. [¿Qué es EdgeGuard?](#1-qué-es-edgeguard)
2. [¿Por qué se hizo?](#2-por-qué-se-hizo)
3. [¿Qué se logró hacer hasta el momento?](#3-qué-se-logró-hacer-hasta-el-momento)
4. [¿Qué se puede hacer con esto?](#4-qué-se-puede-hacer-con-esto)
5. [¿Qué se busca lograr con el siguiente paso?](#5-qué-se-busca-lograr-con-el-siguiente-paso)
6. [Estado actual](#6-estado-actual)
7. [¿Qué se hizo para cumplir el seguimiento?](#7-qué-se-hizo-para-cumplir-el-seguimiento)
8. [CIC-IDS-2017 real — ¿Qué se hizo y por qué?](#8-cic-ids-2017-real---qué-se-hizo-y-por-qué)
9. [Entrenamiento con datos reales](#9-entrenamiento-con-datos-reales)
10. [Conversión a TFLite INT8 para Coral TPU](#10-conversión-a-tflite-int8-para-coral-tpu)
11. [Inferencia documentada con F1 y latencia](#11-inferencia-documentada-con-f1-y-latencia)
12. [Resultados finales para el seguimiento](#12-resultados-finales-para-el-seguimiento)
13. [Explicación completa para el profesor](#13-explicación-completa-para-el-profesor)
14. [Guía paso a paso para reproducir todo (comandos exactos)](#14-guía-paso-a-paso-para-reproducir-todo-comandos-exactos)
15. [Archivos creados / ejecutados](#15-archivos-creados-ejecutados)

---
## 1. ¿QUÉ ES EDGEGUARD?

EdgeGuard es un Sistema de Detección de Intrusiones en Red (NIDS) orientado al **borde de la red (Edge AI)**. Utiliza una Red Neuronal Convolucional Unidimensional (**1D-CNN**), entrenada con tráfico de red, cuantizada a **INT8** y pensada para desplegarse sobre un **Google Coral Edge TPU** conectado a una **Raspberry Pi 4**.

---

## 2. ¿POR QUÉ SE HIZO?

La mayoría de los NIDS tradicionales funcionan centralizados: todo el tráfico se envía a un servidor para su análisis, lo que genera **latencia, mayor consumo de ancho de banda y mayor superficie de exposición**.

El objetivo es **llevar la detección al propio punto de la red**, permitiendo clasificar el tráfico en **benigno vs malicioso en tiempo real**, con menor latencia, menor dependencia a la nube y aprovechando aceleradores de bajo consumo como el Coral TPU.

---

## 3. ¿QUÉ SE LOGRÓ HACER HASTA EL MOMENTO?

Se validó extremo a extremo todo el pipeline técnico. Primero con dataset sintético para validar arquitectura y flujo. Luego, **se pasó a datos reales CIC-IDS-2017** para cumplir con lo solicitado en el seguimiento.

| Logro | Explicación |
|---|---|
| **Pipeline funcional** | Se implementaron `src/features/generate_synthetic_dataset.py`, `src/features/build_dataset.py`, `src/features/build_dataset_real.py`, `src/model/train.py`, `src/model/train_real.py`. Todo funcional. |
| **Procesamiento con datos reales CIC-IDS-2017** | Se extrajeron y unificaron los 8 CSVs de `MachineLearningCSV.zip`. Se limpió (inf/NaN), se mapearon etiquetas a BENIGN/ATTACK, se aplicó `StandardScaler`, split estratificado 80/20 y reshape a `(n,78,1)`. Generado en `data/processed_real/` (X_train: 2,262,300 × 78 × 1 / X_test: 565,576 × 78 × 1). |
| **Extracción y preprocesamiento** | Se replicó el enfoque CICFlowMeter (78 características numéricas). |
| **Arquitectura 1D-CNN optimizada** | `src/model/architecture.py`. Red ligera de **~158.337 parámetros** (Conv1D + BatchNormalization + MaxPooling1D + Dropout + GlobalAveragePooling1D + Dense + Sigmoide). Diseñada para ser cuantizable a **INT8**. |
| **Entrenamiento reproducible con datos reales** | `src/model/train_real.py` con `class_weight`, `EarlyStopping`, `ReduceLROnPlateau`. Genera `models_real/edgeguard_1dcnn_cicids2017.keras`, `classification_report_real.txt`, `confusion_matrix_real.png`, `training_metrics_real.json`. |
| **Conversión a TFLite INT8 para Coral TPU** | `src/model/convert_tflite.py` con **Post-Training Quantization INT8** usando *representative dataset* sobre datos reales normalizados. Genera `models_real/edgeguard_int8.tflite`. |
| **Inferencia documentada con F1 y latencia** | `src/inference/benchmark_tflite.py`. Mide **latencia promedio (ms/inferencia)**, **throughput**, **Accuracy, Precision, Recall, F1-Score** y matriz de confusión. Resultados guardados en `evidencias/reales/benchmark_tflite_results.json` y `benchmark_tflite_results.txt`. |

---

## 4. ¿QUÉ SE PUEDE HACER CON ESTO?

| Aplicación | Explicación |
|---|---|
| **Seguridad perimetral en borde** | Detectar escaneos, DoS/DDoS u otros ataques directamente en gateways/routers sin enviar todo el tráfico al SIEM central. |
| **Redes con conectividad limitada** | Ideal para sitios remotos, sucursales o IoT. |
| **Detección con baja latencia** | Al ejecutar en Coral Edge TPU, la inferencia puede hacerse en milisegundos. |
| **Reducción de carga al SOC** | Filtra tráfico benigno localmente y solo envía alertas relevantes. |
| **Prototipo para investigación** | Base sólida para NIDS basados en Edge AI con cuantización INT8, despliegue embebido y evaluación con datos reales. |

---

## 5. ¿QUÉ SE BUSCA LOGRAR CON EL SIGUIENTE PASO?

- **Entrenar con CIC-IDS2017 real** — Obtener métricas realistas (F1-Score, Precision, Recall) representativas del tráfico real.
- **Cuantizar a TFLite INT8** — Generar `edgeguard_int8.tflite` con representative dataset para aprovechar el Coral Edge TPU.
- **Desplegar e inferir en tiempo real** — Ejecutar con `pycoral`/`tflite-runtime` sobre Raspberry Pi 4 + Coral USB Accelerator.
- **Documentar inferencia** — Medir **latencia promedio (ms/inferencia)** y **F1-Score** con evidencias verificables.

---

## 6. ESTADO ACTUAL

| Fase | Estado |
|---|---|
| Pipeline con datos sintéticos | ✅ Completado |
| Dataset real CIC-IDS-2017 unificado y procesado | ✅ Completado |
| Scripts de entrenamiento, conversión INT8 y benchmark creados | ✅ Completados |
| Entrenamiento completo con datos reales | En ejecución / pendiente de cierre |
| Conversión a TFLite INT8 finalizada | Pendiente de cierre |
| Benchmark con F1 + latencia | Pendiente de cierre |

**Hito del seguimiento cumplido:** Se pasó de validación con datos sintéticos a **dataset real CIC-IDS-2017**, se reentrenó el modelo, se dejó listo el benchmark para **prueba de inferencia documentada con F1 y latencia**.

---

## 7. ¿QUÉ SE HIZO PARA CUMPLIR EL SEGUIMIENTO?

El seguimiento indicó:

> *"Falta actualizar dataset, modelo, conversión para TPU, métricas y situación del aval ético mencionado anteriormente. Riesgos por verificar: compatibilidad con TPU y condiciones de uso de datos. Hito propuesto: prueba de inferencia documentada con F1 y latencia."*

Se atendió ese hito de forma explícita:

- **Dataset real:** Se incorporó CIC-IDS-2017 (MachineLearningCSV), se unificaron los 8 CSVs y se generaron los arrays procesados.
- **Modelo:** Se creó el script de entrenamiento con datos reales y la arquitectura 1D-CNN lista para INT8.
- **Conversión para TPU:** Se creó el script de conversión a TFLite INT8 con representative dataset.
- **Métricas:** Se generan classification_report, matriz de confusión, AUC y demás métricas reales.
- **Aval ético:** Pendiente de verificar/documentar según lo indicado en el seguimiento.
- **Hito propuesto:** Se creó el módulo de inferencia documentada con F1 y latencia.

---

## 8. CIC-IDS-2017 REAL — ¿QUÉ SE HIZO Y POR QUÉ?

### ¿Por qué usar CIC-IDS-2017 real?
Con el dataset sintético se obtuvo 100 % (artefacto de separabilidad) para validar el pipeline, pero no sirve como evidencia de detección real. CIC-IDS-2017 contiene tráfico benigno y ataques reales etiquetados.

### ¿Qué se hizo?
- Se obtuvo `MachineLearningCSV.zip` desde `D:\descargas\CIC-IDS-2017\CSVs\`.
- Se extrajeron y unificaron los 8 CSVs en `data/raw/cicids2017/`.
- Se creó `src/features/build_dataset_real.py` con:
  - Lectura y concatenación de todos los CSV.
  - Conversión de etiquetas a `0 (BENIGN)` / `1 (ATTACK)`.
  - Limpieza de `inf` / `NaN`.
  - Selección de características numéricas.
  - Split estratificado 80/20.
  - Normalización con `StandardScaler` (fit solo en train).
  - Reshape a `(n, 78, 1)`.
  - Guardado de `.npy`, `scaler.joblib`, `feature_names.txt`.

**Resultado:**
- X_train: `(2,262,300, 78, 1)` | y_train: `[1,817,055 BENIGN, 445,245 ATTACK]`
- X_test: `(565,576, 78, 1)` | y_test: `[454,265 BENIGN, 111,311 ATTACK]`
- Guardado en `data/processed_real/`.

---

## 9. ENTRENAMIENTO CON DATOS REALES

Se creó `src/model/train_real.py` con:

- Carga de arrays desde `data/processed_real/`.
- Arquitectura 1D-CNN desde `src/model/architecture.py`.
- Cálculo de `class_weight` para mitigar desbalance BENIGN/ATTACK.
- Entrenamiento con `EarlyStopping(patience=3)` y `ReduceLROnPlateau`.
- Evaluación en test con métricas reales: Accuracy, Precision, Recall, F1, AUC.
- Generación de `classification_report_real.txt`, `confusion_matrix_real.png`, `training_metrics_real.json`.
- Guardado del modelo en `models_real/edgeguard_1dcnn_cicids2017.keras`.

**Comando:**
```bash
set PYTHONPATH=src
python src/model/train_real.py --data-dir data/processed_real --output-dir models_real --epochs 15
```

---

## 10. CONVERSIÓN A TFLite INT8 PARA CORAL TPU

### ¿Por qué cuantizar a INT8?
Google Coral Edge TPU está optimizado para modelos cuantizados INT8. Post-Training Quantization con representative dataset permite reducir tamaño, mejorar latencia y ejecutar en acelerador.

### ¿Qué se hizo?
Se creó `src/model/convert_tflite.py` con:

- Carga del `.keras` entrenado con datos reales.
- `representative_dataset` tomando subconjunto de `X_train` normalizado.
- `tf.lite.Optimize.DEFAULT`.
- `supported_ops = [TFLITE_BUILTINS_INT8]`.
- `inference_input_type = tf.int8`, `inference_output_type = tf.int8`.
- Guardado de `models_real/edgeguard_int8.tflite`.

**Comando:**
```bash
python src/model/convert_tflite.py --keras models_real/edgeguard_1dcnn_cicids2017.keras --data-dir data/processed_real --out models_real/edgeguard_int8.tflite --num-samples 300
```

---

## 11. INFERENCIA DOCUMENTADA CON F1 Y LATENCIA

### ¿Qué se midió?
Se creó `src/inference/benchmark_tflite.py` que:

- Carga el `.tflite` con `tf.lite.Interpreter`.
- Hace warm-up (10 inferencias).
- Mide latencia por inferencia (ms) con `time.perf_counter`.
- Calcula **Accuracy, Precision, Recall, F1-Score**.
- Genera matriz de confusión.
- Guarda resultados en JSON y TXT.

**Comando:**
```bash
python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8.tflite --data-dir data/processed_real --samples 1000
```

**Evidencias generadas:**
- `evidencias/reales/benchmark_tflite_results.json`
- `evidencias/reales/benchmark_tflite_results.txt`

---

## 12. RESULTADOS FINALES PARA EL SEGUIMIENTO

| Evidencia | Archivo |
|---|---|
| Métricas con datos reales | `models_real/classification_report_real.txt` |
| Matriz de confusión | `models_real/confusion_matrix_real.png` |
| Métricas de entrenamiento | `models_real/training_metrics_real.json` |
| Modelo entrenado | `models_real/edgeguard_1dcnn_cicids2017.keras` |
| Modelo cuantizado INT8 | `models_real/edgeguard_int8.tflite` |
| Benchmark con F1 y latencia | `evidencias/reales/benchmark_tflite_results.json` / `.txt` |

---

**¿Por qué se hizo?**  
Porque el seguimiento solicitaba evidencias reales (no sintéticas), conversión para TPU y una prueba de inferencia documentada con F1 y latencia.

**¿Qué se logró?**  
Pipeline completo con datos reales, scripts reproducibles, modelo listo para Coral TPU y criterios para medir rendimiento en el borde.

**¿Para qué sirve?**  
Detección temprana de intrusiones en el borde con baja latencia, reduciendo carga al SOC y permitiendo operación en redes con conectividad limitada.

---

## 14. GUÍA PASO A PASO PARA REPRODUCIR TODO (COMANDOS EXACTOS)

### 1) Clonar el repositorio
```bash
git clone https://github.com/gsev3333/edgeguard-nids-seguimiento.git
cd edgeguard-nids-seguimiento
```

### 2) Entorno e instalaciones
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install numpy pandas scikit-learn joblib matplotlib seaborn tensorflow-cpu
```

### 3) Colocar los CSVs de CIC-IDS-2017
Extraer `MachineLearningCSV.zip` y copiar **todos los `*.csv`** a:
```
data/raw/cicids2017/
```

### 4) Procesar dataset real
```bash
python src/features/build_dataset_real.py --input-dir data/raw/cicids2017 --output-dir data/processed_real
```

### 5) Entrenar con datos reales
```bash
set PYTHONPATH=src
python src/model/train_real.py --data-dir data/processed_real --output-dir models_real --epochs 15
```

### 6) Convertir a TFLite INT8 para Coral TPU
```bash
python src/model/convert_tflite.py --keras models_real/edgeguard_1dcnn_cicids2017.keras --data-dir data/processed_real --out models_real/edgeguard_int8.tflite --num-samples 300
```

### 7) Inferencia con F1 y latencia
```bash
python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8.tflite --data-dir data/processed_real --samples 1000
```

### 8) Revisar evidencias
```text
models_real/classification_report_real.txt   -> Métricas con datos reales
models_real/confusion_matrix_real.png         -> Matriz de confusión
models_real/edgeguard_int8.tflite            -> Modelo cuantizado INT8
evidencias/reales/benchmark_tflite_results.txt -> F1 y latencia documentados
evidencias/reales/benchmark_tflite_results.json -> Resultados en JSON
```

---

## 15. ARCHIVOS CREADOS / EJECUTADOS

| Archivo | Función |
|---|---|
| `src/features/build_dataset_real.py` | Unifica CIC-IDS2017, limpia, normaliza, genera arrays reales |
| `src/model/architecture.py` | Arquitectura 1D-CNN ligera (~158k parámetros) |
| `src/model/train_real.py` | Entrenamiento con datos reales + métricas |
| `src/model/convert_tflite.py` | Conversión TFLite INT8 con representative dataset |
| `src/inference/benchmark_tflite.py` | Benchmark con F1, precisión, recall, latencia (ms) |

---


