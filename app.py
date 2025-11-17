from flask import Flask
from flask import render_template, request
import Reg_Logis as ReLogistica
import Relineal
import pandas as pd
import knn
import os

app = Flask(__name__)

# Variable global para almacenar las conclusiones actuales
conclusiones_actuales = None

@app.route('/')
def inicio():
    return render_template('inicio.html')

@app.route('/menu')
def menu():
    return render_template('menu.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/index3')
def index3():
    return render_template('index3.html')

@app.route('/index4')
def index4():
    return render_template('index4.html')

@app.route('/LR', methods = ["GET", "POST"])
def LR():
    calculateResult = None
    if request.method == "POST":
        try:
            altitud = float(request.form["altitud"])
            frecuencia = float(request.form["frecuencia"])
            calculateResult = Relineal.CalculateOxygen(altitud, frecuencia)
            
            import time
            time.sleep(0.1)
            
            Relineal.save_plot(altitud, frecuencia, calculateResult)
            
        except ValueError:
            return "Por favor ingrese valores numéricos válidos"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return render_template("rl.html", result = calculateResult)

@app.route('/conceptos')
def conceptos():
    return render_template('conceptos.html')

# Regresión Logística
ReLogistica.evaluate()

@app.route('/conceptos_reg_logistica')
def conceptos_reg_logistica():
    return render_template('conceptos_reg_logistica.html')

# Cargar datos desde el archivo CSV
try:
    data = pd.read_csv('./DataSheet/data.csv', delimiter=';')
except FileNotFoundError:
    print("Error: El archivo data.csv no se encontró.")
    exit(1)
except Exception as e:
    print(f"Error al cargar datos: {e}")
    exit(1)

@app.route('/ejercicio_reg_logistica', methods=['GET', 'POST'])
def ejercicio_reg_logistica():
    result = None
    if request.method == 'POST':
        try:
            edad = float(request.form['edad'])
            tiempo = float(request.form['tiempo'])
            tipo = request.form['tipo'].lower()
            visitas = float(request.form['visitas'])

            entrada = pd.DataFrame([{
                "edad_mascota": edad,
                "tiempo_adopcion": tiempo,
                "visitas_recibidas": visitas,
                "tipo_mascota": tipo
            }])

            entrada = pd.get_dummies(entrada, columns=["tipo_mascota"], drop_first=True)

            for col in ReLogistica.x.columns:
                if col not in entrada.columns:
                    entrada[col] = 0
            entrada = entrada[ReLogistica.x.columns]

            features = entrada.values[0]
            etiqueta, probabilidad = ReLogistica.predict_label(features)

            result = {
                "etiqueta": etiqueta,
                "probabilidad": probabilidad
            }

        except ValueError:
            result = {"error": "Por favor ingrese valores válidos"}
        except Exception as e:
            result = {"error": f"Error: {str(e)}"}

    return render_template('ejercicio_reg_logistica.html', result=result)

#===============   knn   =====================#

@app.route('/TiposAlgoritmos')
def tipos_algoritmos():
    return render_template('TiposAlgoritmos.html')

@app.route('/ejercicio_knn', methods=['GET', 'POST'])
def ejercicio_knn():
    global conclusiones_actuales
    
    metrics = None
    pred = None
    prob = None

    if request.method == 'POST':
        if 'train' in request.form:
            try:
                # Verificar si el archivo de datos existe y tiene contenido nuevo
                datos_nuevos = verificar_datos_nuevos()
                
                # Entrenar modelo y obtener resultados completos
                resultado_completo = knn.entrenar_modelo()
                
                metrics = {
                    "accuracy": resultado_completo["accuracy"],
                    "report": resultado_completo["report"],
                    "classes": resultado_completo["classes"]
                }
                
                # Guardar conclusiones globalmente
                conclusiones_actuales = resultado_completo["conclusiones"]
                
                # Guardar timestamp del entrenamiento
                with open("static/ultimo_entrenamiento.txt", "w") as f:
                    f.write(str(pd.Timestamp.now()))
                
                pred = request.form.get('pred')
                prob = request.form.get('prob')
                
            except Exception as e:
                metrics = {'error': f'Error entrenando: {e}'}

        elif 'predict' in request.form:
            try:
                edad = float(request.form['edad'])
                peso = float(request.form['peso'])
                frecuencia = float(request.form['frecuencia'])
                actividad_num = int(request.form['actividad'])
                threshold = float(request.form.get('threshold', 0.5))

                actividad_map = {1: "Baja", 2: "Media", 3: "Alta"}
                actividad = actividad_map.get(actividad_num, "Media")

                features = {
                    "Edad": edad,
                    "Peso": peso,
                    "Frecuencia": frecuencia,
                    "NivelActividad": actividad
                }

                pred, prob = knn.predict_label(features, threshold)

            except Exception as e:
                pred = f"Error: {e}"
                prob = None
    else:
        # En GET, intentar cargar conclusiones existentes
        if conclusiones_actuales is None and os.path.exists("static/knn_model.pkl"):
            try:
                # Generar conclusiones a partir del modelo existente
                conclusiones_actuales = knn.generar_conclusiones_desde_modelo_existente()
            except:
                conclusiones_actuales = None

    return render_template(
        'ejercicio_knn.html',
        metrics=metrics,
        pred=pred,
        prob=prob,
        conclusiones=conclusiones_actuales
    )

def verificar_datos_nuevos():
    """Verifica si los datos han cambiado desde el último entrenamiento"""
    archivo_datos = "DataSheet/Knn_data.csv"
    archivo_entrenamiento = "static/ultimo_entrenamiento.txt"
    
    if not os.path.exists(archivo_entrenamiento):
        return True
    
    if not os.path.exists(archivo_datos):
        return False
    
    try:
        # Comparar timestamps
        mod_time_datos = os.path.getmtime(archivo_datos)
        with open(archivo_entrenamiento, "r") as f:
            ultimo_entrenamiento = pd.Timestamp(f.read().strip())
        
        return mod_time_datos > ultimo_entrenamiento.timestamp()
    except:
        return True

@app.route('/rl_conceptos')
def rl_conceptos():
    return render_template('rl_conceptos.html')

@app.route('/rl_ejercicio')
def rl_ejercicio():
    return render_template('rl_ejercicio.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)