from flask import Flask
from flask import render_template, request
import Relineal


app = Flask(__name__)

# Casos de uso de Machine Learning

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



# Regresión Lineal

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
@app.route('/conceptos_reg_logistica')
    
def conceptos_reg_logistica():
    return render_template('conceptos_reg_logistica.html')

@app.route('/ejercicio_reg_logistica')
def ejercicio_reg_logistica():
    return render_template('ejercicio_reg_logistica.html')





if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Especifica el puerto
