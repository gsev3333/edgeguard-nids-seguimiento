# EdgeGuard

EdgeGuard es un proyecto académico que busca reconocer conexiones de red que podrían ser maliciosas y avisar a una persona responsable. La idea es que el análisis se realice cerca de la red observada, en un equipo pequeño, en vez de enviar todo el tráfico a un servidor central.

Por ahora, EdgeGuard **clasifica archivos de conexiones ya preparados**. Todavía no escucha una red en vivo, no genera alertas operativas y no bloquea tráfico. El uso futuro propuesto es un laboratorio o segmento piloto autorizado, con observación pasiva.

## Qué problema busca atender

Una red produce muchas conexiones y revisarlas una por una es difícil. EdgeGuard explora si un modelo de inteligencia artificial puede ayudar a señalar cuáles conexiones merecen revisión. Una conexión se representa con un conjunto de datos resumidos; el modelo devuelve una probabilidad y la clasifica como:

- `BENIGN`: el registro está marcado como benigno.
- `ATTACK`: el registro tiene una etiqueta distinta de `BENIGN`.

El modelo actual usa una red neuronal convolucional de una dimensión (CNN 1D). En términos sencillos, aprende patrones entre 78 valores de cada conexión. La salida es binaria: no identifica todavía qué familia de ataque encontró.

## Qué queremos lograr y cómo se implementará

La meta es construir un prototipo que reciba una copia autorizada de conexiones, prepare sus 78 valores de la misma manera que los datos de entrenamiento, los clasifique y guarde un aviso consultivo que pueda revisar una persona. La propuesta no incluye bloquear equipos o conexiones automáticamente.

| Etapa | Qué se hará | Estado |
|---|---|---|
| 1. Preparar y entender los datos | Limpiar CIC-IDS-2017, revisar los valores y separar datos para entrenar y evaluar. | Realizada para la línea base. |
| 2. Entrenar y medir el modelo | Aprender a distinguir registros `BENIGN` y `ATTACK`; revisar errores y ajustar el umbral de aviso. | Realizada en archivos preparados. |
| 3. Preparar el modelo pequeño | Convertirlo a formato TFLite INT8 para reducir su tamaño y revisar la inferencia en CPU. | Realizada en CPU. |
| 4. Comprobar Google Coral | Compilar el modelo y medirlo con un acelerador Coral conectado. | Pendiente: aún no se dispone del dispositivo ni de evidencia de ejecución en TPU. |
| 5. Conectar una fuente de red | Crear la captura pasiva y el extractor que convierta conexiones reales en los mismos 78 valores. | Pendiente. Requiere permiso y validación de cada valor. |
| 6. Registrar avisos y evaluar | Guardar resultados para revisión humana y medir su utilidad en un piloto controlado. | Pendiente de integración, autorización y criterios acordados. |

El recorrido esperado sería: **copia autorizada del tráfico → resumen de cada conexión → clasificación de EdgeGuard → registro consultivo → revisión humana**. La captura, el resumen en vivo y el registro de avisos aún deben desarrollarse. Hasta entonces, el prototipo solo demuestra la clasificación de datos preparados.

## Dónde se aplicaría

El primer lugar propuesto es un laboratorio académico o un segmento pequeño de la organización de práctica. La red entregaría una copia del tráfico mediante un puerto espejo o un TAP; EdgeGuard observaría esa copia sin quedar en medio de la conexión. La universidad, la sede, el segmento y las personas responsables todavía deben confirmarse.

Antes de usar tráfico de una organización se necesita autorización escrita, definir quién puede ver los datos, cuánto tiempo se guardan y cómo se protegen. Las pruebas explicadas abajo se realizan con CIC-IDS-2017. No conectes el prototipo a una red institucional sin aprobación.

## Estado comprobado

| Parte | Resultado actual |
|---|---|
| Datos CIC-IDS-2017 | Se procesaron ocho archivos CSV: 2.827.876 registros limpios y 78 valores por registro. |
| Modelo CNN 1D | Entrenado para separar `BENIGN` y `ATTACK`. |
| Evaluación | Revisada en 565.576 registros reservados para prueba. Hay reportes y una matriz de errores. |
| Umbral de clasificación | Se comparó el umbral inicial con uno elegido usando datos de validación. |
| Modelo TFLite INT8 | Conversión verificada con entradas y salidas enteras; tamaño registrado de 187.040 bytes. |
| Inferencia en CPU | Medida en Windows 11 con 1.000 registros. El resultado describe ese PC y esa muestra. |
| Google Coral Edge TPU | No probado: no hay un Coral disponible ni registro de compilación. |
| Captura en vivo y avisos | No implementados. |

## Pruebas y evidencias

Esta sección permite comprobar el proyecto paso a paso y conservar evidencia de lo que ocurrió. Hay dos niveles:

1. **Pruebas rápidas:** revisan que el código se pueda cargar, que el modelo produzca una salida y que los cálculos de conversión de TFLite funcionen. No requieren datos ni Coral.
2. **Prueba completa:** procesa el conjunto CIC-IDS-2017, entrena el modelo, lo convierte y mide la inferencia en CPU. Toma más tiempo y requiere descargar el conjunto.

Una prueba de código aprobada no demuestra por sí sola que el detector funcione bien en una red. Las métricas del conjunto de prueba tampoco demuestran detección en vivo.

### A. Preparar el entorno en Windows

Se recomienda Windows 11 y Python 3.12. Abre PowerShell en la carpeta del repositorio y ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si el equipo no reconoce `py`, crea el entorno con `python -m venv .venv` y continúa con los demás comandos. El modelo se entrenó en CPU; el tiempo dependerá del computador.

### B. Ejecutar las pruebas rápidas

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

La primera orden revisa errores de escritura en los archivos Python. La segunda ejecuta seis pruebas pequeñas: forma de salida del modelo, valores de probabilidad y conversión de valores hacia y desde el formato entero de TFLite. El resultado esperado termina en `Ran 6 tests` y `OK`. Estas pruebas no usan CIC-IDS-2017, no entrenan el modelo y no prueban Coral.

### C. Preparar el conjunto para la prueba completa

Descarga CIC-IDS-2017 desde el [sitio del Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html). Usa el paquete `MachineLearningCSV.zip` con los archivos CSV del conjunto. En el repositorio, crea la carpeta y guarda allí el ZIP con ese nombre:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
```

Luego prepara los datos:

```powershell
python src/features/build_dataset_real.py --input-zip "data/raw/MachineLearningCSV.zip" --output-dir data/processed_real --expected-features 78
```

**Qué comprobar:** el resumen debe indicar los archivos procesados, 78 valores por conexión, la cantidad de registros y la separación de entrenamiento y prueba. El archivo `data/processed_real/dataset_summary.json` guarda esa evidencia. Si aparecen menos archivos o cambia la cantidad de valores, detén el proceso y revisa que descargaste el paquete correcto; no continúes con un conjunto distinto sin documentarlo.

El script también guarda los arreglos numéricos y el escalador en `data/processed_real/`. Esos archivos se generan localmente y no se suben a GitHub.

### D. Entrenar y revisar el modelo

```powershell
python src/model/train_real.py --data-dir data/processed_real --output-dir models_real --epochs 15 --batch-size 512 --seed 42
```

El entrenamiento puede detenerse antes de las 15 épocas si deja de mejorar. Al finalizar deben existir `models_real/edgeguard_1dcnn_cicids2017.keras`, `training_metrics_real.json`, `classification_report_real.txt` y `confusion_matrix_real.png`.

Abre el reporte y la matriz: muestran cuántas conexiones detectó y cuáles confundió. La matriz diferencia falsas alarmas (conexiones benignas marcadas como ataque) de ataques omitidos (ataques que el modelo no señaló). Conserva los archivos con la fecha y versión de código de la ejecución.

### E. Elegir el umbral de aviso

El umbral indica qué probabilidad mínima convierte una conexión en aviso. Un umbral bajo suele señalar más ataques, pero también puede generar más falsas alarmas. El script lo elige con una parte de los datos de entrenamiento reservada para validación y después informa el resultado en la prueba:

```powershell
python src/model/calibrate_threshold.py --keras models_real/edgeguard_1dcnn_cicids2017.keras --data-dir data/processed_real --output models_real/threshold_calibration_real.json --validation-fraction 0.1 --target-attack-recall 0.95
```

**Qué comprobar:** el archivo `models_real/threshold_calibration_real.json` debe incluir el umbral seleccionado, el resultado en validación y las métricas de prueba. El valor 0,95 es una condición usada para elegir el umbral en validación; no garantiza ese mismo resultado en datos nuevos.

### F. Convertir el modelo y probarlo en CPU

Convierte el modelo a TFLite INT8 usando 300 ejemplos para estimar la escala de conversión:

```powershell
python src/model/convert_tflite.py --keras models_real/edgeguard_1dcnn_cicids2017.keras --data-dir data/processed_real --out models_real/edgeguard_int8.tflite --num-samples 300 --seed 42
```

Comprueba que se generen `models_real/edgeguard_int8.tflite` y `models_real/edgeguard_int8.json`, y que el informe indique entrada y salida `int8`. Esta conversión **no significa** que el modelo ya funcione en Coral.

Para guardar cada medición en una carpeta nueva, corre estas órdenes en la misma ventana de PowerShell:

```powershell
$evidencia = "evidencias/reales/prueba_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$umbral = (Get-Content "models_real/threshold_calibration_real.json" | ConvertFrom-Json).threshold

python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8.tflite --data-dir data/processed_real --samples 1000 --warmup 10 --seed 42 --device cpu --threshold 0.5 --out-dir "$evidencia/cpu_umbral_0_5"
python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8.tflite --data-dir data/processed_real --samples 1000 --warmup 10 --seed 42 --device cpu --threshold $umbral --out-dir "$evidencia/cpu_umbral_calibrado"
```

Cada carpeta debe contener un archivo `.json` y uno `.txt`. Ahí quedan las métricas, la latencia, el equipo y la matriz de errores para poder repetir la comparación. La medición incluye preparar la entrada e invocar el modelo, pero no incluye capturar tráfico ni construir conexiones a partir de paquetes.

Como referencia de una ejecución ya realizada en este proyecto, el modelo Keras obtuvo AUC ROC 0,989 y recall de ataques 0,661 con umbral 0,5. Con umbral calibrado, el recall en la prueba completa fue 0,948, pero produjo 24.161 falsas alarmas y omitió 5.787 ataques. En una muestra de 1.000 registros, la conversión INT8 en CPU obtuvo recall 0,710 y latencia media de 0,0430 ms. Las cifras pueden variar con las versiones de software, el equipo y los ejemplos; registra los resultados de tu propia ejecución.

### G. Prueba con Coral (pendiente de hardware)

Esta prueba solo se puede realizar cuando haya un Google Coral conectado y esté instalado el software compatible con esa plataforma. La compilación se ejecuta en un sistema Linux compatible con `edgetpu_compiler`:

```bash
mkdir -p evidencias/reales/coral
edgetpu_compiler -s --out_dir models_real models_real/edgeguard_int8.tflite 2>&1 | tee evidencias/reales/coral/compilacion.txt
```

Revisa el archivo `compilacion.txt` y confirma qué operaciones se asignaron al acelerador. Después, en el equipo con Coral, ejecuta:

```bash
python src/inference/benchmark_tflite.py --tflite models_real/edgeguard_int8_edgetpu.tflite --data-dir data/processed_real --samples 1000 --warmup 10 --seed 42 --device coral --threshold 0.0597913712 --out-dir evidencias/reales/coral
```

La evidencia mínima es el registro del compilador y los archivos `benchmark_coral_results.json` y `benchmark_coral_results.txt`. No marques esta prueba como aprobada si el dispositivo no aparece en el resultado o si la ejecución fue en CPU.

### H. Qué guardar como evidencia

Para cada ejecución conserva:

- fecha, equipo, versión de Python y versión de las dependencias;
- `dataset_summary.json` y la cantidad de archivos de entrada;
- métricas, reporte y matriz de confusión del modelo;
- `threshold_calibration_real.json`;
- `edgeguard_int8.json`;
- resultados `.json` y `.txt` de cada benchmark;
- si se prueba Coral, el registro de compilación y el nombre exacto del dispositivo.

No publiques datos de una red real ni archivos que contengan direcciones, nombres u otra información sensible. Para un piloto institucional, primero acuerda qué evidencia se puede compartir.

## Qué significan los resultados y qué falta

El modelo ya distingue registros del conjunto de investigación y puede ejecutar una inferencia TFLite en CPU. Eso demuestra una etapa experimental, pero no que el sistema detecte incidentes en una organización. Las falsas alarmas deben ser revisadas por una persona y los ataques omitidos siguen siendo importantes.

Antes de llegar al piloto se necesita:

1. conseguir y probar el Coral en la plataforma objetivo;
2. construir el capturador pasivo y el extractor de conexiones;
3. confirmar que el extractor entrega exactamente los mismos 78 valores, en el mismo orden y escala;
4. generar avisos consultivos y guardar registros con controles de acceso;
5. definir con la institución las métricas de aceptación, el permiso, la retención y quién revisará los avisos;
6. hacer una prueba limitada y comparar sus resultados con una medición inicial.

Por eso, EdgeGuard es hoy un **prototipo de clasificación offline** y un candidato a piloto académico. No está listo para producción ni para bloquear tráfico.

## Archivos principales

| Ruta | Para qué sirve |
|---|---|
| `src/features/build_dataset_real.py` | Prepara el conjunto de datos. |
| `src/model/architecture.py` | Define la CNN 1D. |
| `src/model/train_real.py` | Entrena el modelo y guarda métricas. |
| `src/model/calibrate_threshold.py` | Elige el umbral con datos de validación. |
| `src/model/convert_tflite.py` | Convierte el modelo a TFLite INT8. |
| `src/inference/benchmark_tflite.py` | Mide la ejecución en CPU o Coral. |
| `tests/` | Pruebas rápidas de forma de salida y conversión numérica. |
| `docs/PLAN_APLICACION_Y_VIABILIDAD.md` | Explica el propósito, el escenario propuesto, las etapas y la viabilidad. |

El ZIP de datos, los arreglos y los modelos generados se guardan localmente, no en GitHub. Los reportes pequeños de referencia sí están versionados para que se puedan consultar sin volver a entrenar.
