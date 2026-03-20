import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json, datetime, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

train_df = pd.read_csv(os.path.join(DATA_DIR, "Train_data.csv"))

for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    joblib.dump(le, os.path.join(BASE_DIR, f"{col}_classes.pkl"))

X = train_df.drop("class", axis=1)
y = train_df["class"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=100, random_state=42, 
    n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)

y_pred  = model.predict(X_val)
y_proba = model.predict_proba(X_val)
anomaly_idx = list(model.classes_).index('anomaly')
y_bin   = (y_val == 'anomaly').astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred, labels=['normal','anomaly'])
tn, fp, fn, tp = cm.ravel()

metadata = {
    "sklearn_version"    : __import__('sklearn').__version__,
    "trained_at"         : datetime.datetime.now().isoformat(),
    "n_estimators"       : model.n_estimators,
    "n_features"         : model.n_features_in_,
    "feature_names"      : list(X.columns),
    "classes"            : list(model.classes_),
    "accuracy"           : round(accuracy_score(y_val, y_pred), 6),
    "f1_anomaly"         : round(f1_score(y_val, y_pred, pos_label='anomaly'), 6),
    "roc_auc"            : round(roc_auc_score(y_bin, y_proba[:,anomaly_idx]), 6),
    "top_features"       : list(__import__('pandas').Series(
                               model.feature_importances_, 
                               index=X.columns).nlargest(10).index),
    "confusion_matrix"   : cm.tolist(),
    "false_positive_rate": round(fp/(fp+tn)*100, 4),
    "false_negative_rate": round(fn/(fn+tp)*100, 4),
    "training_rows"      : len(X_train),
    "val_rows"           : len(X_val)
}

joblib.dump(model, os.path.join(BASE_DIR, "ids_model.pkl"))
with open(os.path.join(BASE_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Model retrained and saved with sklearn", metadata['sklearn_version'])
print("Accuracy:", metadata['accuracy'])
print("F1:", metadata['f1_anomaly'])
print("ROC-AUC:", metadata['roc_auc'])
