# knn.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
def entrenar_modelo(k=5, forzar_retrenamiento=False):
    try:
        # Si no forzar retrenamiento y existe modelo, usar el existente
        if not forzar_retrenamiento and os.path.exists("static/knn_model.pkl"):
            print("Cargando modelo existente...")
            conclusiones = generar_conclusiones_desde_modelo_existente()
            if conclusiones:
                # Cargar métricas básicas del modelo existente
                model = joblib.load("static/knn_model.pkl")
                df = load_data()
                preprocessor, X_train, X_test, y_train, y_test = prepare_data(df)
                y_pred = model.predict(X_test)
                acc = round(accuracy_score(y_test, y_pred), 4)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                return {
                    "accuracy": acc,
                    "report": report,
                    "classes": list(model.classes_),
                    "conclusiones": conclusiones
                }
        
        print("Entrenando nuevo modelo...")
        df = load_data()
        print(f"Datos cargados: {df.shape}")
        print(f"Columnas: {df.columns.tolist()}")
        
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

        # Guardar modelo
        os.makedirs("static", exist_ok=True)
        joblib.dump(model, "static/knn_model.pkl")

        # Generar matriz de confusión
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión K-NN")
        plt.tight_layout()
        plt.savefig("static/cm_knn.png")
        plt.close()

        print("Generando conclusiones...")
        conclusiones = generar_conclusiones_completas(df, X_train, X_test, y_train, y_test, acc, cm, model)
        print("Conclusiones generadas exitosamente")
        
        return {
            "accuracy": acc,
            "report": report,
            "classes": list(model.classes_),
            "conclusiones": conclusiones
        }
        
    except Exception as e:
        print(f"Error en entrenar_modelo: {e}")
        import traceback
        traceback.print_exc()
        return {
            "accuracy": 0,
            "report": {},
            "classes": [],
            "conclusiones": {
                "comparativa_algoritmos": {"resultados": {}, "mejor_algoritmo": "N/A", "peor_algoritmo": "N/A", "analisis": ["Error en análisis"]},
                "analisis_problema": ["Error generando análisis"],
                "impacto_umbral_desbalance": {"balance": "Error", "distribucion": {}, "ratio_balance": 0, "impacto_umbral": {}},
                "mejoras_especificas": ["Error generando mejoras"],
                "resumen_final": ["Error en resumen"]
            }
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

# ---------------------------
# 5. ANÁLISIS COMPARATIVO Y CONCLUSIONES COMPLETAS
# ---------------------------
def generar_conclusiones_completas(df, X_train, X_test, y_train, y_test, knn_acc, knn_cm, knn_model):
    """
    Genera conclusiones completas basadas en los puntos específicos solicitados
    """
    conclusiones = {}
    
    # 1. COMPARACIÓN DE ALGORITMOS
    conclusiones['comparativa_algoritmos'] = comparar_algoritmos_completo(X_train, X_test, y_train, y_test, knn_model, knn_acc)
    
    # 2. ANÁLISIS DE CARACTERÍSTICAS DEL PROBLEMA
    conclusiones['analisis_problema'] = analizar_caracteristicas_detallado(df, knn_acc, knn_cm, X_train.shape[1])
    
    # 3. IMPACTO DEL UMBRAL Y DESBALANCE
    conclusiones['impacto_umbral_desbalance'] = analizar_umbral_desbalance(knn_model, X_test, y_test, df['TipoRutina'])
    
    # 4. IDEAS DE MEJORA ESPECÍFICAS
    conclusiones['mejoras_especificas'] = generar_mejoras_especificas(df, knn_acc, conclusiones['comparativa_algoritmos'])
    
    # 5. CONCLUSIONES FINALES RESUMIDAS
    conclusiones['resumen_final'] = generar_resumen_final(conclusiones)
    
    return conclusiones

def comparar_algoritmos_completo(X_train, X_test, y_train, y_test, knn_model, knn_acc):
    """Compara KNN con otros algoritmos y analiza rendimiento"""
    algoritmos = {
        'K-NN': KNeighborsClassifier(n_neighbors=5),
        'Regresión Logística': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
        'SVM Lineal': SVC(kernel='linear', random_state=RANDOM_STATE, probability=True, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    }
    
    resultados = {}
    analisis_rendimiento = []
    preprocessor = knn_model.named_steps['preprocessor']
    
    for nombre, modelo in algoritmos.items():
        try:
            if nombre == 'K-NN':
                acc = knn_acc
            else:
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', modelo)
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            
            resultados[nombre] = round(acc, 4)
            
        except Exception as e:
            resultados[nombre] = f"Error: {str(e)}"
    
    # ANÁLISIS DE RENDIMIENTO
    resultados_validos = {k: v for k, v in resultados.items() if isinstance(v, (int, float))}
    
    if resultados_validos:
        mejor_algo = max(resultados_validos, key=resultados_validos.get)
        peor_algo = min(resultados_validos, key=resultados_validos.get)
        
        # Análisis de no linealidad
        if 'K-NN' in resultados_validos and 'Regresión Logística' in resultados_validos:
            diff_knn_lr = resultados_validos['K-NN'] - resultados_validos['Regresión Logística']
            if diff_knn_lr > 0.05:
                analisis_rendimiento.append("🔹 K-NN supera a Regresión Logística → datos con relaciones NO LINEALES")
            elif diff_knn_lr < -0.05:
                analisis_rendimiento.append("🔹 Regresión Logística supera a K-NN → relaciones más LINEALES")
            else:
                analisis_rendimiento.append("🔹 Rendimientos similares → mezcla de patrones lineales y no lineales")
        
        # Análisis de complejidad
        if mejor_algo == 'Random Forest' and resultados_validos['Random Forest'] > knn_acc + 0.1:
            analisis_rendimiento.append("🔹 Random Forest mejor → datos con interacciones complejas entre features")
    else:
        mejor_algo = "N/A"
        peor_algo = "N/A"
    
    return {
        'resultados': resultados,
        'mejor_algoritmo': mejor_algo,
        'peor_algoritmo': peor_algo,
        'analisis': analisis_rendimiento
    }

def analizar_caracteristicas_detallado(df, knn_acc, knn_cm, num_features):
    """Análisis detallado de características del problema"""
    analisis = []
    
    # TAMAÑO DE MUESTRA
    n_samples = len(df)
    if n_samples < 50:
        analisis.append("📉 Tamaño de muestra MUY PEQUEÑO: alto riesgo de sobreajuste")
    elif n_samples < 100:
        analisis.append("📊 Tamaño de muestra limitado: K-NN puede tener alta varianza")
    else:
        analisis.append("📈 Tamaño de muestra adecuado para K-NN")
    
    # DIMENSIONALIDAD
    if num_features > 10:
        analisis.append("🌀 ALTA dimensionalidad: K-NN sufre de 'maldición de la dimensionalidad'")
    elif num_features <= 3:
        analisis.append("📐 BAJA dimensionalidad: favorable para K-NN")
    else:
        analisis.append("📏 Dimensionalidad moderada: K-NN funciona adecuadamente")
    
    # RUIDO EN LOS DATOS
    numeric_vars = ['Edad', 'Peso', 'Frecuencia']
    try:
        coef_variacion = df[numeric_vars].std() / df[numeric_vars].mean()
        ruido_promedio = coef_variacion.mean()
        
        if ruido_promedio > 0.5:
            analisis.append("🎯 ALTO ruido en datos: K-NN sensible a outliers y variabilidad")
        elif ruido_promedio > 0.3:
            analisis.append("🎯 Ruido moderado: K-NN maneja adecuadamente la variabilidad")
        else:
            analisis.append("🎯 BAJO ruido: datos consistentes, favorable para K-NN")
    except:
        analisis.append("🎯 No se pudo calcular el nivel de ruido en los datos")
    
    # COMPLEJIDAD DE FRONTERAS
    if knn_acc > 0.9:
        analisis.append("🎯 FRONTERAS BIEN DEFINIDAS: alta separabilidad entre clases")
    elif knn_acc > 0.7:
        analisis.append("🎯 Fronteras moderadamente complejas: algunas superposiciones")
    else:
        analisis.append("🎯 FRONTERAS MUY COMPLEJAS: alta superposición entre clases")
    
    return analisis

def analizar_umbral_desbalance(modelo, X_test, y_test, series_clases):
    """Análisis del impacto del umbral y desbalance de clases"""
    analisis = {}
    
    # ANÁLISIS DE DESBALANCE
    counts = series_clases.value_counts()
    if len(counts) > 0:
        balance_ratio = counts.min() / counts.max()
        
        if balance_ratio > 0.7:
            analisis['balance'] = "✅ BIEN BALANCEADO"
        elif balance_ratio > 0.4:
            analisis['balance'] = "⚠️ MODERADAMENTE DESBALANCEADO"
        else:
            analisis['balance'] = "❌ FUERTEMENTE DESBALANCEADO"
        
        analisis['distribucion'] = counts.to_dict()
        analisis['ratio_balance'] = round(balance_ratio, 3)
    else:
        analisis['balance'] = "❌ ERROR: No hay clases"
        analisis['distribucion'] = {}
        analisis['ratio_balance'] = 0
    
    # IMPACTO DEL UMBRAL
    try:
        probas = modelo.predict_proba(X_test)
        umbrales = [0.3, 0.5, 0.7, 0.9]
        resultados_umbral = {}
        
        for umbral in umbrales:
            y_pred_umbral = modelo.classes_[np.argmax(probas, axis=1)]
            # Aplicar umbral de confianza
            confianzas = np.max(probas, axis=1)
            mascara_confianza = confianzas >= umbral
            
            if mascara_confianza.sum() > 0:
                acc = accuracy_score(y_test[mascara_confianza], y_pred_umbral[mascara_confianza])
                resultados_umbral[umbral] = {
                    'accuracy': round(acc, 3),
                    'porcentaje_rechazadas': round((1 - mascara_confianza.mean()) * 100, 1),
                    'instancias_aceptadas': mascara_confianza.sum()
                }
        
        analisis['impacto_umbral'] = resultados_umbral
    except:
        analisis['impacto_umbral'] = {}
    
    return analisis

def generar_mejoras_especificas(df, knn_acc, comparativa):
    """Genera ideas de mejora específicas basadas en el análisis"""
    mejoras = []
    n_samples = len(df)
    
    # MEJORAS POR DESBALANCE
    if "DESBALANCEADO" in str(comparativa.get('analisis', [''])):
        mejoras.extend([
            "⚖️ Usar class_weight='balanced' en algoritmos sensibles",
            "🔄 Aplicar SMOTE para oversampling de clases minoritarias",
            "📊 Usar F1-score como métrica principal en lugar de accuracy"
        ])
    
    # MEJORAS POR TAMAÑO DE MUESTRA
    if n_samples < 100:
        mejoras.extend([
            "📈 Recolectar MÁS DATOS: idealmente 200+ instancias",
            "🎯 Aplicar cross-validation robusto (10-fold)",
            "🔍 Usar técnicas de data augmentation si es posible"
        ])
    
    # MEJORAS PARA K-NN ESPECÍFICAS
    mejoras.extend([
        "🎯 Optimizar K con GridSearchCV (probando k=3,5,7,9,11)",
        "📏 Probar diferentes distancias (Manhattan para features heterogéneos)",
        "🏋️‍♂️ Usar K-NN ponderado por distancia",
        "🧹 Aplicar selección de características con SelectKBest"
    ])
    
    # MEJORAS DE FEATURES
    mejoras.extend([
        "🔧 Ingeniería de características: crear interacciones entre variables",
        "📐 Normalización específica por dominio de conocimiento",
        "🎯 Análisis de correlaciones para eliminar features redundantes"
    ])
    
    # CALIBRACIÓN
    mejoras.extend([
        "📊 Calibrar probabilidades con CalibratedClassifierCV",
        "🎯 Ajustar umbrales de decisión por clase",
        "📈 Usar curva precision-recall para optimizar trade-offs"
    ])
    
    return mejoras

def generar_resumen_final(conclusiones):
    """Genera un resumen ejecutivo final"""
    resumen = []
    
    # Rendimiento de algoritmos
    comp = conclusiones['comparativa_algoritmos']
    resumen.append(f"🏆 MEJOR ALGORITMO: {comp['mejor_algoritmo']}")
    resumen.append(f"📊 PEOR ALGORITMO: {comp['peor_algoritmo']}")
    
    # Análisis de características
    analisis = conclusiones['analisis_problema']
    for punto in analisis[:2]:  # Primeros 2 puntos más importantes
        resumen.append(punto)
    
    # Impacto del desbalance
    balance_info = conclusiones['impacto_umbral_desbalance']
    resumen.append(f"⚖️ BALANCE: {balance_info['balance']}")
    
    return resumen

def generar_conclusiones_desde_modelo_existente():
    """Genera conclusiones a partir del modelo existente sin reentrenar"""
    try:
        # Cargar modelo existente
        model_path = "static/knn_model.pkl"
        if not os.path.exists(model_path):
            return None
            
        model = joblib.load(model_path)
        
        # Cargar datos actuales
        df = load_data()
        
        # Preparar datos para evaluación
        preprocessor, X_train, X_test, y_train, y_test = prepare_data(df)
        
        # Evaluar modelo existente
        y_pred = model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred), 4)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        
        # Generar conclusiones
        conclusiones = generar_conclusiones_completas(df, X_train, X_test, y_train, y_test, acc, cm, model)
        
        return conclusiones
        
    except Exception as e:
        print(f"Error generando conclusiones desde modelo existente: {e}")
        return None

def cargar_y_evaluar_modelo_existente():
    """Carga el modelo existente y lo evalúa con los datos actuales"""
    try:
        model_path = "static/knn_model.pkl"
        if not os.path.exists(model_path):
            return None, None, None, None
            
        model = joblib.load(model_path)
        df = load_data()
        
        # Usar el preprocessor del modelo cargado
        preprocessor = model.named_steps['preprocessor']
        
        # Preparar datos
        X = df.drop('TipoRutina', axis=1)
        y = df['TipoRutina']
        
        # Aplicar transformaciones
        X_processed = preprocessor.transform(X)
        
        # Hacer predicciones
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        
        return model, df, acc, y_pred
        
    except Exception as e:
        print(f"Error cargando modelo existente: {e}")
        return None, None, None, None


# Plantilla HTML (templates/knn.html)
HTML_TEMPLATE = """
{% extends "base.html" %}
{% block title %}Caso práctico K-NN{% endblock %}
{% block content %}
<div class="container mt-4 bg-light ">

  <h2 class="mb-3 card-header bg-primary text-white text-center rounded-top-4">
    Clasificación con K-Nearest Neighbors (K-NN)</h2>
  <p>
    Este caso práctico recomienda un tipo de rutina fitness (<strong>Cardio</strong>, 
    <strong>Fuerza</strong> o <strong>Mixta</strong>) para nuevos usuarios.  
    <br>Variables usadas:
  </p>
  <ul>
    <li><strong>Edad</strong> (numérica)</li>
    <li><strong>Peso</strong> (numérica)</li>
    <li><strong>Frecuencia cardíaca en reposo</strong> (numérica)</li>
    <li><strong>Nivel de actividad física</strong> (categórica: 1, 2, 3)</li>
  </ul>
  <p>Clase positiva: <strong>Tipo de rutina sugerida</strong>.</p>

  <!-- Entrenamiento -->
  <hr>
  <h4>Entrenamiento del modelo</h4>
  <form method="POST">
    <button class="btn btn-primary" name="train" value="1">Entrenar Modelo</button>
    <button class="btn btn-warning" name="forzar_train" value="1" style="margin-left: 10px;">
        Forzar Re-entrenamiento
    </button>
  </form>

  {% if metrics %}
    <div class="mt-4">
      <h5>Exactitud:</h5>
      <div class="card p-3 mb-3">
        <span style="font-size:1.5rem; font-weight:bold;">{{ metrics.accuracy }}</span>
      </div>

      <h5>Reporte de clasificación:</h5>
      <table class="table table-bordered table-sm mt-2">
        <thead>
          <tr>
            <th>Clase</th>
            <th>Precisión</th>
            <th>Recall</th>
            <th>F1-score</th>
            <th>Soporte</th>
          </tr>
        </thead>
        <tbody>
          {% for clase, datos in metrics.report.items() if clase in metrics.classes %}
          <tr>
            <td>{{ clase }}</td>
            <td>{{ "%.2f"|format(datos.precision) }}</td>
            <td>{{ "%.2f"|format(datos.recall) }}</td>
            <td>{{ "%.2f"|format(datos['f1-score']) }}</td>
            <td>{{ datos.support }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>

      <h5>Matriz de confusión:</h5>
      <img src="{{ url_for('static', filename='cm_knn.png') }}" class="img-fluid mt-2" alt="Matriz de confusión">
    </div>
  {% endif %}

  <!-- Conclusiones -->
  {% if conclusiones %}
  <div class="mt-5">
    <h4>Análisis y Conclusiones</h4>
    
    <!-- Comparativa de Algoritmos -->
    <div class="card mb-3">
      <div class="card-header bg-info text-white">
        <h5>Comparativa de Algoritmos</h5>
      </div>
      <div class="card-body">
        <table class="table table-striped">
          <thead>
            <tr>
              <th>Algoritmo</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {% for algo, acc in conclusiones.comparativa_algoritmos.resultados.items() %}
            <tr>
              <td>{{ algo }}</td>
              <td>{{ acc }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        <p><strong>Mejor algoritmo:</strong> {{ conclusiones.comparativa_algoritmos.mejor_algoritmo }}</p>
        <p><strong>Peor algoritmo:</strong> {{ conclusiones.comparativa_algoritmos.peor_algoritmo }}</p>
        <ul>
          {% for analisis in conclusiones.comparativa_algoritmos.analisis %}
          <li>{{ analisis }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>

    <!-- Análisis del Problema -->
    <div class="card mb-3">
      <div class="card-header bg-warning">
        <h5>Análisis del Problema</h5>
      </div>
      <div class="card-body">
        <ul>
          {% for punto in conclusiones.analisis_problema %}
          <li>{{ punto }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>

    <!-- Impacto del Umbral y Desbalance -->
    <div class="card mb-3">
      <div class="card-header bg-danger text-white">
        <h5>Impacto del Umbral y Desbalance</h5>
      </div>
      <div class="card-body">
        <p><strong>Balance de clases:</strong> {{ conclusiones.impacto_umbral_desbalance.balance }}</p>
        <p><strong>Ratio de balance:</strong> {{ conclusiones.impacto_umbral_desbalance.ratio_balance }}</p>
        <p><strong>Distribución:</strong> {{ conclusiones.impacto_umbral_desbalance.distribucion }}</p>
        
        <h6>Impacto del umbral de confianza:</h6>
        <table class="table table-sm">
          <thead>
            <tr>
              <th>Umbral</th>
              <th>Accuracy</th>
              <th>% Rechazadas</th>
              <th>Instancias Aceptadas</th>
            </tr>
          </thead>
          <tbody>
            {% for umbral, datos in conclusiones.impacto_umbral_desbalance.impacto_umbral.items() %}
            <tr>
              <td>{{ umbral }}</td>
              <td>{{ datos.accuracy }}</td>
              <td>{{ datos.porcentaje_rechazadas }}%</td>
              <td>{{ datos.instancias_aceptadas }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Mejoras Específicas -->
    <div class="card mb-3">
      <div class="card-header bg-success text-white">
        <h5>Mejoras Específicas</h5>
      </div>
      <div class="card-body">
        <ul>
          {% for mejora in conclusiones.mejoras_especificas %}
          <li>{{ mejora }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>

    <!-- Resumen Final -->
    <div class="card mb-3">
      <div class="card-header bg-primary text-white">
        <h5>Resumen Final</h5>
      </div>
      <div class="card-body">
        <ul>
          {% for punto in conclusiones.resumen_final %}
          <li>{{ punto }}</li>
          {% endfor %}
        </ul>
      </div>
    </div>
  </div>
  {% endif %}

  <!-- Predicción -->
  <hr>
  <h4>Predicción para nuevo usuario</h4>
  <form method="POST" class="bg-light p-3 rounded">
    <div class="row">
      <div class="col-md-3">
        <label>Edad:</label>
        <input type="number" name="edad" class="form-control" required>
      </div>
      <div class="col-md-3">
        <label>Peso (kg):</label>
        <input type="number" step="0.1" name="peso" class="form-control" required>
      </div>
      <div class="col-md-3">
        <label>Frecuencia cardíaca:</label>
        <input type="number" name="frecuencia" class="form-control" required>
      </div>
      <div class="col-md-3">
        <label>Nivel Actividad (1-3):</label>
        <select name="nivel_actividad" class="form-control" required>
          <option value="1">1 - Baja</option>
          <option value="2">2 - Media</option>
          <option value="3">3 - Alta</option>
        </select>
      </div>
    </div>
    <button type="submit" class="btn btn-success mt-3" name="predict" value="1">Predecir Rutina</button>
  </form>

  {% if prediction %}
  <div class="alert alert-info mt-3">
    <h5>Resultado de la predicción:</h5>
    <p><strong>Rutina recomendada:</strong> {{ prediction.clase }}</p>
    <p><strong>Confianza:</strong> {{ prediction.confianza }}</p>
  </div>
  {% endif %}

</div>
{% endblock %}
"""

# Función para guardar la plantilla HTML
def guardar_plantilla_html():
    """Guarda la plantilla HTML en el directorio templates"""
    os.makedirs("templates", exist_ok=True)
    with open("templates/knn.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)
    print("Plantilla HTML guardada en templates/knn.html")

if __name__ == "__main__":
    # Guardar la plantilla HTML cuando se ejecute el script
    guardar_plantilla_html()
    print("Código completo generado exitosamente!")