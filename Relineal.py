import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df= pd.DataFrame(data)

# Definimos Variables
x = df[["Study Hours"]]
y = df[["Final Grade"]]

model = LinearRegression()
model.fit(x,y)

# Predecimos
def CalculateGrade(hours: float):
 """Predecimos la nota final a partir de las horas de estudio"""

 result = model.predict([[hours]])[0][0]
 return round (float(result), 2) 

# Graficar (para usar en Flask)
def save_plot(new_x=None, new_y=None):
    plt.figure(figsize=(6,4))
    plt.scatter(x, y, color="blue", label="Datos reales")
    plt.plot(x, model.predict(x), color="red", label="Regresión lineal")

    # Si hay un nuevo dato, lo marcamos en la gráfica
    if new_x is not None and new_y is not None:
        plt.scatter(new_x, new_y, color="green", s=100, label="Nuevo dato")

    plt.xlabel("Horas de estudio")
    plt.ylabel("Nota final")
    plt.legend()
    plt.title("Regresión Lineal: Horas de estudio vs Nota final")
    plt.grid(True)
    plt.savefig("static/images/regresion.png")
    plt.close()
