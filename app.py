from flask import Flask
from flask import render_template, request
import Reg_Logis as ReLogistica
import Relineal
import pandas as pd


app = Flask(__name__)


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
            
            # Pequeña pausa para evitar conflictos
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
            # Obtener datos del formulario
            edad = float(request.form['edad'])
            tiempo = float(request.form['tiempo'])
            tipo = request.form['tipo'].lower()  # perro, gato, roedor
            visitas = float(request.form['visitas'])

            # Crear DataFrame con la fila de entrada
            entrada = pd.DataFrame([{
                "edad_mascota": edad,
                "tiempo_adopcion": tiempo,
                "visitas_recibidas": visitas,
                "tipo_mascota": tipo
            }])

            # Generar variables dummy igual que en entrenamiento
            entrada = pd.get_dummies(entrada, columns=["tipo_mascota"], drop_first=True)

            # Asegurar que tenga las mismas columnas que el modelo
            for col in ReLogistica.x.columns:
                if col not in entrada.columns:
                    entrada[col] = 0
            entrada = entrada[ReLogistica.x.columns]  # mismo orden

            # Convertir a lista para pasarlo a predict_label
            features = entrada.values[0]

            # Llamar a la función de predicción de Reg_Logis
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





if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Especifica el puerto

