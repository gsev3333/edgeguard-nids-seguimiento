import os, json, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.architecture import build_1d_cnn, compile_model

DATA_DIR = sys.argv[sys.argv.index('--data-dir')+1] if '--data-dir' in sys.argv else 'data/processed_real'
OUT_DIR = sys.argv[sys.argv.index('--output-dir')+1] if '--output-dir' in sys.argv else 'models_real'
EPOCHS = int(sys.argv[sys.argv.index('--epochs')+1]) if '--epochs' in sys.argv else 15

os.makedirs(OUT_DIR, exist_ok=True)
X_train = np.load(os.path.join(DATA_DIR,'X_train.npy'))
X_test = np.load(os.path.join(DATA_DIR,'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR,'y_train.npy'))
y_test = np.load(os.path.join(DATA_DIR,'y_test.npy'))
input_shape = X_train.shape[1:]
model = build_1d_cnn(input_shape=input_shape)
model = compile_model(model)
# class weights
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = {int(c): float(w) for c,w in zip(classes,cw)}
callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)]
history = model.fit(X_train,y_train,validation_split=0.1,epochs=EPOCHS,batch_size=512,class_weight=class_weight,callbacks=callbacks,verbose=1)
y_pred = (model.predict(X_test, batch_size=1024) > 0.5).astype(int).ravel()
y_prob = model.predict(X_test, batch_size=1024).ravel()
report = classification_report(y_test,y_pred, target_names=['BENIGN','ATTACK'], output_dict=True)
cm = confusion_matrix(y_test,y_pred)
auc = roc_auc_score(y_test,y_prob)
metrics = {'report':report,'roc_auc':auc,'history':{k:[float(v) for v in history.history[k]] for k in history.history}}
with open(os.path.join(OUT_DIR,'training_metrics_real.json'),'w') as f:
    json.dump(metrics,f,indent=2)
with open(os.path.join(OUT_DIR,'classification_report_real.txt'),'w') as f:
    f.write(classification_report(y_test,y_pred, target_names=['BENIGN','ATTACK']))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN','ATTACK'], yticklabels=['BENIGN','ATTACK'])
plt.title('Confusion Matrix - EdgeGuard CIC-IDS2017')
plt.savefig(os.path.join(OUT_DIR,'confusion_matrix_real.png'), dpi=150, bbox_inches='tight')
plt.close()
model.save(os.path.join(OUT_DIR,'edgeguard_1dcnn_cicids2017.keras'))
print('SAVED')
