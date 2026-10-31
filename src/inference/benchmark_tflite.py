import os, json, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

DATA_DIR = sys.argv[sys.argv.index('--data-dir')+1] if '--data-dir' in sys.argv else 'data/processed_real'
TFLITE = sys.argv[sys.argv.index('--tflite')+1] if '--tflite' in sys.argv else 'models_real/edgeguard_int8.tflite'
OUT_DIR = sys.argv[sys.argv.index('--out-dir')+1] if '--out-dir' in sys.argv else 'evidencias/reales'
SAMPLES = int(sys.argv[sys.argv.index('--samples')+1]) if '--samples' in sys.argv else 1000

os.makedirs(OUT_DIR, exist_ok=True)
X_test = np.load(os.path.join(DATA_DIR,'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR,'y_test.npy'))
# use subset
idx = np.random.choice(len(X_test), size=min(SAMPLES,len(X_test)), replace=False)
X_test_s = X_test[idx]
y_test_s = y_test[idx]

interpreter = tf.lite.Interpreter(model_path=TFLITE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# quantization
in_scale, in_zero = input_details[0]['quantization']
out_scale, out_zero = output_details[0]['quantization']

# warmup
for i in range(10):
    x = X_test_s[:1]
    if in_scale > 0:
        xq = (x / in_scale + in_zero).astype(np.int8)
    else:
        xq = x.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], xq)
    interpreter.invoke()

latencies = []
preds = []
import time
for i in range(len(X_test_s)):
    x = X_test_s[i:i+1]
    if in_scale > 0:
        xq = (x / in_scale + in_zero).astype(np.int8)
    else:
        xq = x.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], xq)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    latencies.append((t1-t0)*1000.0)
    out = interpreter.get_tensor(output_details[0]['index'])[0][0]
    if out_scale > 0:
        prob = (out - out_zero) * out_scale
    else:
        prob = float(out)
    preds.append(1 if prob > 0.5 else 0)

preds = np.array(preds)
y_test_s = y_test_s.astype(int)
acc = accuracy_score(y_test_s,preds)
prec = precision_score(y_test_s,preds,zero_division=0)
rec = recall_score(y_test_s,preds,zero_division=0)
f1 = f1_score(y_test_s,preds,zero_division=0)
cm = confusion_matrix(y_test_s,preds)
lat_avg = float(np.mean(latencies))
res = {'samples':int(len(X_test_s)),'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'latency_ms_avg':lat_avg,'latency_ms_std':float(np.std(latencies)),'confusion_matrix':cm.tolist()}
with open(os.path.join(OUT_DIR,'benchmark_tflite_results.json'),'w') as f:
    json.dump(res,f,indent=2)
with open(os.path.join(OUT_DIR,'benchmark_tflite_results.txt'),'w') as f:
    f.write('EdgeGuard TFLite INT8 Benchmark\n')
    f.write(f'Samples: {len(X_test_s)}\n')
    f.write(f'Accuracy: {acc:.6f}\n')
    f.write(f'Precision: {prec:.6f}\n')
    f.write(f'Recall: {rec:.6f}\n')
    f.write(f'F1-Score: {f1:.6f}\n')
    f.write(f'Latency avg (ms): {lat_avg:.4f}\n')
    f.write(f'Latency std (ms): {np.std(latencies):.4f}\n')
    f.write(f'Confusion Matrix:\n{cm}\n')
print(json.dumps(res))
