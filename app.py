from flask import Flask
from flask import render_template, request
import Relineal


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
        hours = float(request.form["hours"])
        calculateResult = Relineal.CalculateGrade(hours)

        Relineal.save_plot()# para que nos muestre la grafica
    return render_template("rl.html", result = calculateResult)

@app.route('/conceptos')
def conceptos():
    return render_template('conceptos.html')


if __name__ == '__main__':
    app.run(debug=True)