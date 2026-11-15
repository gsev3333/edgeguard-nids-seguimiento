# EdgeGuard — NIDS basado en Edge AI

Prototipo de clasificación binaria de flujos de red con una CNN 1D y datos CIC-IDS-2017. El despliegue objetivo es una Raspberry Pi 4 con Google Coral Edge TPU.

> **Estado verificado al 15 de noviembre de 2026:** se procesaron los ocho CSV del conjunto; se entrenó y evaluó la CNN; se calibró un umbral usando la validación del entrenamiento; se convirtió el modelo a TFLite con entrada y salida INT8; y se midió inferencia CPU en Windows 11. No hay Google Coral disponible para esta ejecución, así que no se afirma compilación ni inferencia Edge TPU. La captura de tráfico en vivo y las alertas tampoco están implementadas.

## Resumen del sistema

EdgeGuard clasifica cada flujo ya extraído como:

- 0: BENIGN.
- 1: ATTACK (cualquier etiqueta no vacía distinta de BENIGN).

El flujo técnico reproducible es:

1. Leer y limpiar los ocho CSV del paquete MachineLearningCSV.zip.
2. Separar entrenamiento y prueba con estratificación y semilla 42.
3. Ajustar el escalador solo con entrenamiento para evitar fuga de información.
4. Entrenar y evaluar una CNN 1D.
5. Seleccionar un umbral operativo con la porción de validación del entrenamiento.
6. Convertir el modelo a TFLite con cuantización entera INT8 y medir su ejecución en CPU.
7. Como trabajo futuro, compilar para Edge TPU y medir con el delegado Coral en hardware real.

Una conversión TFLite INT8 no demuestra por sí misma que el modelo haya sido compilado para Edge TPU ni que se esté ejecutando en ese acelerador.

## Alcance y estado de los componentes

| Componente | Estado comprobado |
|---|---|
| Preprocesamiento CIC-IDS-2017 | Ejecutado sobre los ocho CSV; 2.827.876 registros limpios y 78 características. |
| Arquitectura 1D-CNN | Implementada y entrenada para clasificación binaria. |
| Evaluación del modelo | Ejecutada en 565.576 registros de prueba; reporte, matriz y métricas guardados. |
| Calibración de umbral | Script reproducible; umbral escogido en validación para priorizar recall de ataques. |
| Conversión a TFLite INT8 | Ejecutada; entrada y salida verificadas como int8; archivo de 187.040 bytes. |
| Benchmark TFLite CPU | Ejecutado en Windows 11 con 1.000 registros y dos umbrales. |
| Compilación e inferencia Coral | Pendiente: falta compilador compatible y acelerador Google Coral físico. |
| Captura de paquetes, extracción de flujos y alertas | No implementadas; el sistema recibe flujos ya tabulados. |
| Revisión o aval ético institucional | Debe confirmarse con el escenario de práctica y la Universidad. |

El repositorio contiene preprocesamiento, arquitectura, entrenamiento y benchmark TFLite CPU. Esta actualización incorpora el convertidor INT8, la calibración de umbral, el benchmark CPU/Coral, los resultados pequeños reproducibles y las instrucciones de instalación. El ZIP original, los arrays y los pesos del modelo permanecen fuera de Git para no duplicar datos y artefactos generables.

## Datos y preprocesamiento

El archivo utilizado fue MachineLearningCSV.zip, con ocho archivos CSV de CIC-IDS-2017. El script puede leerlos desde el ZIP sin descomprimir todo el paquete.

El preprocesamiento:

- Une los CSV y normaliza espacios de encabezados y etiquetas.
- Convierte BENIGN en 0 y todas las demás etiquetas no vacías en ATTACK.
- Convierte las características a valores numéricos y elimina filas con valores ausentes o infinitos.
- Excluye metadatos no numéricos y comprueba la cantidad esperada de características.
- Realiza una partición estratificada 80/20 con semilla 42.
- Ajusta StandardScaler solo con la partición de entrenamiento y lo aplica a la de prueba.
- Conserva el orden de características y da forma a cada ejemplo como (78, 1).
- Guarda los arrays, el escalador, los nombres de características y el resumen JSON.

Resultados reproducidos:

| Partición | Registros | BENIGN | ATTACK | Forma de características |
|---|---:|---:|---:|---|
| Entrenamiento | 2.262.300 | 1.817.055 | 445.245 | (2.262.300, 78, 1) |
| Prueba | 565.576 | 454.265 | 111.311 | (565.576, 78, 1) |
| Total limpio | 2.827.876 | 2.271.320 | 556.556 | 78 características |

El resumen conserva los conteos de las etiquetas originales. Algunas etiquetas de ataques web contienen el carácter de reemplazo Unicode en los CSV; la clasificación binaria no cambia porque todas se agrupan como ATTACK. Esto deberá resolverse si se amplía a clasificación por familia de ataque.

Artefactos generados localmente:

- data/processed_real/dataset_summary.json
- data/processed_real/X_train.npy y X_test.npy
- data/processed_real/y_train.npy y y_test.npy
- data/processed_real/scaler.joblib
- data/processed_real/feature_names.txt

El ZIP, los arrays, el escalador y los pesos binarios del modelo no se suben. Ocupan más espacio, requieren los datos originales para reproducirse y deben respetar las condiciones de distribución. Para que el avance sea verificable, sí se versionan reportes, métricas y benchmarks compactos seleccionados; el resto de esos directorios permanece excluido por .gitignore.

## Modelo, entrenamiento y evaluación

La CNN tiene tres bloques Conv1D con normalización por lotes, MaxPooling y Dropout, seguidos por GlobalAveragePooling, una capa densa y salida sigmoide. La entrada tiene forma (78, 1). El entrenamiento aplica pesos por clase, EarlyStopping y ReduceLROnPlateau. Se solicitaron 15 épocas, pero la parada temprana concluyó después de 4 y restauró los mejores pesos de validación.

Métricas del conjunto de prueba con el umbral convencional 0,5:

| Métrica | BENIGN | ATTACK |
|---|---:|---:|
| Precisión | 0,923 | 0,970 |
| Recall | 0,995 | 0,661 |
| F1 | 0,958 | 0,786 |

Exactitud: 0,929. AUC ROC: 0,989. Matriz de confusión en orden [BENIGN, ATTACK]:

    [[451964, 2301],
     [37715, 73596]]

El umbral 0,5 detecta pocos falsos positivos, pero omite 37.715 de los 111.311 ataques del conjunto de prueba. Por ello, la exactitud global no debe usarse como único criterio de calidad para un NIDS.

### Calibración del umbral

Se añadió src/model/calibrate_threshold.py. Reproduce la separación de validación de Keras: toma el 10 % final de los arrays de entrenamiento, que Keras reserva antes de barajar cuando se usa validation_split=0.1. Escoge el umbral con mayor F1 en esa validación, sujeto a un recall de ataques de al menos 0,95. El conjunto de prueba no interviene en la selección.

El umbral seleccionado fue 0,059791. Al evaluar una vez en el conjunto de prueba:

| Métrica | BENIGN | ATTACK |
|---|---:|---:|
| Precisión | 0,987 | 0,814 |
| Recall | 0,947 | 0,948 |
| F1 | 0,966 | 0,876 |

Exactitud: 0,947. AUC ROC: 0,989. Matriz de confusión:

    [[430104, 24161],
     [5787, 105524]]

La calibración reduce los ataques omitidos de 37.715 a 5.787, pero aumenta los falsos positivos de 2.301 a 24.161. El recall de prueba fue 0,948, ligeramente menor que el objetivo 0,95 usado en validación; la selección en validación no garantiza alcanzar exactamente esa cifra en datos no vistos. El umbral debe tratarse como un punto operativo provisional y revisarse con el escenario de práctica y sus costos de error.

Archivos de entrenamiento y evaluación:

- models_real/edgeguard_1dcnn_cicids2017.keras
- models_real/training_metrics_real.json
- models_real/classification_report_real.txt
- models_real/confusion_matrix_real.png
- models_real/threshold_calibration_real.json

Los archivos JSON/TXT de métricas y el gráfico de matriz de confusión se incluyen como evidencia compacta. Los modelos .keras/.tflite y los arrays no se incluyen; se pueden regenerar con los comandos de reproducción.

## Conversión TFLite y benchmark CPU

El convertidor usa 300 ejemplos deterministas de X_train como conjunto representativo. Fuerza operaciones enteras TFLite y comprueba que tanto la entrada como la salida sean INT8. El archivo generado es models_real/edgeguard_int8.tflite; su tamaño es 187.040 bytes. Los parámetros de cuantización de entrada/salida están en models_real/edgeguard_int8.json.

Resultados del benchmark CPU sobre 1.000 ejemplos de prueba seleccionados con semilla 42, después de 10 inferencias de calentamiento. La medición incluye cuantización del registro, invocación y lectura/descuantización de salida; no incluye extracción de flujos ni lectura de disco.

| Umbral | Exactitud | Precisión ATTACK | Recall ATTACK | F1 ATTACK | AUC ROC | Latencia media | P95 | Inferencias/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,5 | 0,918 | 0,984 | 0,614 | 0,756 | 0,955 | 0,0383 ms | 0,0415 ms | 26.117 |
| 0,059791 | 0,931 | 0,942 | 0,710 | 0,810 | 0,955 | 0,0430 ms | 0,0622 ms | 23.237 |

La latencia corresponde a este equipo Windows 11 y al runtime TensorFlow Lite con delegado XNNPACK CPU; no representa rendimiento en Raspberry Pi ni Coral. La muestra de 1.000 registros sirve para una comprobación reproducible del pipeline, no sustituye la evaluación completa de 565.576 registros.

Evidencias compactas del benchmark:

- evidencias/reales/threshold_0_5/benchmark_cpu_results.json y benchmark_cpu_results.txt
- evidencias/reales/benchmark_cpu_results.json y benchmark_cpu_results.txt (umbral calibrado)

## Reproducción

### Entorno utilizado

- Windows 11, Python 3.12.10.
- TensorFlow 2.21.0 ejecutado en CPU; TensorFlow nativo en Windows no utilizó GPU.
- Dependencias Python instaladas desde requirements.txt en el entorno local .venv.

Preparar el entorno en Windows:

    py -3.12 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

En Linux/macOS, activa el entorno con:

    source .venv/bin/activate

### Procesar y entrenar

    python src/features/build_dataset_real.py --input-zip "ruta/al/MachineLearningCSV.zip" --output-dir data/processed_real --expected-features 78
    python src/model/train_real.py --data-dir data/processed_real --output-dir models_real --epochs 15 --batch-size 512 --seed 42
    python src/model/calibrate_threshold.py --keras models_real/edgeguard_1dcnn_cicids2017.keras --data-dir data/processed_real --output models_real/threshold_calibration_real.json --validation-fraction 0.1 --target-attack-recall 0.95

### Convertir y medir en CPU

    python src/model/convert_tflite.py --keras models_real/edgeguard_1dcnn_cicids2017.keras --data-dir data/processed_real --out models_real/edgeguard_int8.tflite --num-samples 300 --seed 42
    python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8.tflite --data-dir data/processed_real --samples 1000 --warmup 10 --seed 42 --device cpu --threshold 0.0597913712

El benchmark permite seleccionar otro umbral con --threshold. Para comparar umbrales sobre la misma muestra se fija la misma semilla y número de registros.

### Compilar y medir en Google Coral

Hace falta un sistema Linux x86-64 compatible con edgetpu_compiler para compilar el modelo y una plataforma compatible con PyCoral y el acelerador conectado para ejecutar la inferencia. La conversión INT8 desde Windows no reemplaza estos pasos.

En el entorno compatible, la compilación se ejecuta así:

    edgetpu_compiler -s --out_dir models_real models_real/edgeguard_int8.tflite

Después, revisar el informe del compilador para conocer las operaciones delegadas y las que, en su caso, queden en CPU. Ejecutar el benchmark únicamente con el archivo compilado y el dispositivo conectado:

    python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8_edgetpu.tflite --data-dir data/processed_real --samples 1000 --warmup 10 --seed 42 --device coral --threshold 0.0597913712

Las guías oficiales de Coral explican la [compilación Edge TPU](https://coral.ai/docs/edgetpu/compiler/) y la [inferencia TFLite con Python](https://coral.ai/docs/edgetpu/tflite-python/). A la fecha de esta ejecución no hay un Coral disponible y WSL no está instalado; por eso esos resultados siguen pendientes.

## Limitaciones y pendientes reales

1. Ejecutar la compilación Edge TPU, guardar el registro y medir con un Coral conectado; comparar sobre los mismos ejemplos, semilla y umbral.
2. Mejorar la sensibilidad del detector y examinar los falsos positivos por tipo de tráfico; el recall de ATTACK observado con el umbral calibrado fue 0,948.
3. Incorporar, si lo exige el alcance institucional, captura de paquetes, extracción de flujos con las mismas 78 características y orden, normalización y generación de alertas. Hoy se clasifican registros preparados fuera de línea.
4. Evaluar generalización por día o escenario y por familia de ataque. La partición aleatoria estratificada puede contener patrones de los mismos días en entrenamiento y prueba.
5. Confirmar condiciones de uso y redistribución del dataset, y el estado de aval o revisión ética con la institución.
6. Conservar las versiones, especificaciones del equipo objetivo y evidencias de cada ejecución para una comparación válida CPU/Coral.

## Información técnica para preparar el futuro informe APA

Este README organiza evidencia técnica para elaborar después el informe de intervención; no inventa datos institucionales que aún no se han proporcionado. El código y los resultados permiten sustentar la denominación provisional, describir el problema técnico del prototipo, explicar el método implementado y reportar indicadores verificables. El desempeño observado no demuestra que el proyecto esté desplegado en una red real.

Para redactar los componentes del informe APA todavía se necesitan:

- Matriz DOFA y acciones de intervención del Informe del marco institucional.
- Descripción del escenario de práctica, ubicación, cobertura espacial y necesidades verificadas.
- Beneficiarios autorizados y población a la que se dirige el proyecto.
- Responsables, roles, fechas aprobadas y duración de la práctica.
- Recursos humanos, materiales, costos reales y presupuesto institucional.
- Estado y soporte documental de aval o revisión ética, y condiciones de uso del dataset.
- Instrumento de diagnóstico y medición posterior para comparar la situación inicial y los resultados.
- Completar la revisión bibliográfica con las fuentes iniciales listadas abajo, las fuentes que indique la Universidad y citas integradas en cada afirmación del informe.

Indicadores técnicos ya observables: archivos procesados, filas limpias, número y orden de características, conteos por clase, precisión/recall/F1/AUC, matriz de confusión, tipo de cuantización, latencia y throughput CPU. Indicadores pendientes: compilación para Edge TPU, operaciones delegadas, medición en Coral y cualquier comparación con tráfico real. Los resultados y evidencias se encuentran en los archivos listados en este documento.

La matriz de confusión se expresa como [[verdaderos BENIGN, falsos ATTACK], [ataques clasificados BENIGN, verdaderos ATTACK]].

### Revisión bibliográfica preliminar

Sharafaldin et al. (2018) describen la creación de CIC-IDS-2017, los escenarios de captura, las familias de ataque y la extracción de características de flujo. Singh et al. (2021) presentan otro clasificador 1D-CNN evaluado con CIC-IDS-2017 y reportan precisión, recall y F1. Estos trabajos apoyan la pertinencia del enfoque, pero sus resultados no son directamente comparables con EdgeGuard porque pueden diferir el preprocesamiento, la partición y el etiquetado.

La evaluación debe considerar las limitaciones del benchmark: Engelen et al. (2021) examinan el proceso de construcción de CIC-IDS2017, y Lanvin et al. (2023) documentan errores posibles de captura y etiquetado que pueden cambiar la evaluación. Por eso se reportan falsos negativos y falsos positivos, y se recomienda evaluar por día o escenario antes de plantear un uso en una red real. La fundamentación final deberá ampliarse con bibliografía reciente sobre NIDS en Edge AI y ejecución en Edge TPU.

### Referencias iniciales (formato APA 7)

Canadian Institute for Cybersecurity. (s. f.). *Intrusion detection evaluation dataset (CIC-IDS2017).* University of New Brunswick. https://www.unb.ca/cic/datasets/ids-2017.html

Engelen, G., Rimmer, V., & Joosen, W. (2021). Troubleshooting an intrusion detection dataset: The CICIDS2017 case study. En *2021 IEEE Security and Privacy Workshops (SPW)* (pp. 7–12). IEEE. https://doi.org/10.1109/SPW53761.2021.00009

Google AI Edge. (s. f.). *Post-training quantization.* https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization

Google Coral. (s. f.). *Edge TPU compiler.* https://coral.ai/docs/edgetpu/compiler/

Lanvin, M., Gimenez, P. F., Han, Y., Majorczyk, F., Mé, L., & Totel, É. (2023). Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes. En S. Kallel, M. Jmaiel, A. Hadj Kacem, M. Zulkernine, F. Cuppens, & N. Cuppens (Eds.), *Risks and security of internet and systems: 17th international conference, CRiSIS 2022, revised selected papers* (Vol. 13857, pp. 18–33). Springer. https://doi.org/10.1007/978-3-031-31108-6_2

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. En *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)* (pp. 108–116). SCITEPRESS. https://doi.org/10.5220/0006639801080116

Singh, K., Mahajan, A., & Mansotra, V. (2021). 1D-CNN based model for classification and analysis of network attacks. *International Journal of Advanced Computer Science and Applications, 12*(11). https://doi.org/10.14569/IJACSA.2021.0121169
