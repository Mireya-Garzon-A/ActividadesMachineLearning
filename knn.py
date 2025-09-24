# knn.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Semilla para reproducibilidad
RANDOM_STATE = 42

# ---------------------------
# 1. Carga de datos
# ---------------------------
def load_data(path="DataSheet/Knn_data.csv"):
    df = pd.read_csv(path, delimiter=';')
    df.columns = ['Edad', 'Peso', 'Frecuencia', 'NivelActividad', 'TipoRutina']
    return df

# ---------------------------
# 2. Preprocesamiento y split
# ---------------------------
def prepare_data(df):
    X = df.drop('TipoRutina', axis=1)
    y = df['TipoRutina']

    cat_cols = ['NivelActividad']
    num_cols = ['Edad', 'Peso', 'Frecuencia']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    return preprocessor, X_train, X_test, y_train, y_test

# ---------------------------
# 3. Entrenamiento y evaluación
# ---------------------------
def entrenar_modelo(k=5):
    df = load_data()
    preprocessor, X_train, X_test, y_train, y_test = prepare_data(df)

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=k))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred), 4)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    # Asegurar carpeta
    os.makedirs("static", exist_ok=True)
    joblib.dump(model, "static/knn_model.pkl")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión K-NN")
    plt.tight_layout()
    plt.savefig("static/cm_knn.png")
    plt.close()

    return {
        "accuracy": acc,
        "report": report,
        "classes": list(model.classes_)
    }

# ---------------------------
# 4. Predicción individual
# ---------------------------
def predict_label(features_dict, threshold=0.5):
    """
    features_dict debe tener claves: Edad, Peso, Frecuencia, NivelActividad
    NivelActividad debe coincidir con los valores del entrenamiento (ej. 'Baja', 'Media', 'Alta')
    """
    model_path = "static/knn_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Entrena primero.")

    model = joblib.load(model_path)
    input_df = pd.DataFrame([features_dict])
    probas = model.predict_proba(input_df)[0]
    idx = np.argmax(probas)
    return model.classes_[idx], round(probas[idx], 4)

