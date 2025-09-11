import matplotlib
matplotlib.use('Agg')  # ✅ Usa backend no interactivo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de ejemplo para altitud y cantidad de oxígeno en sangre
data = {
    "Altitud": [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 
                500, 1500, 2000, 3000, 1000, 2500, 3500, 4000, 0, 2000],
    "Frecuencia Respiratoria": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                13, 17, 19, 23, 15, 21, 25, 27, 11, 18],
    "Saturacion Oxigeno (%)": [98, 97, 96, 95, 94, 93, 92, 91, 90, 89,
                               97, 95, 94, 92, 96, 93, 91, 90, 99, 94]
}

df = pd.DataFrame(data)

# Definimos Variables
x = df[["Altitud", "Frecuencia Respiratoria"]]
y = df[["Saturacion Oxigeno (%)"]]

model = LinearRegression()
model.fit(x, y)

# Función para predecir SpO₂
def CalculateOxygen(altitud: float, frecuencia: float):
    """Predice la saturación de oxígeno en sangre (SpO₂ %)"""
    result = model.predict([[altitud, frecuencia]])[0][0]
    return round(float(result), 2)

# Función para graficar
def save_plot(altitud=None, frecuencia=None, spo2=None):
    try:
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Altitud vs SpO₂
        ax1.scatter(df["Altitud"], df["Saturacion Oxigeno (%)"], color="blue", label="Datos reales", alpha=0.6)
        altitud_range = np.linspace(df["Altitud"].min(), df["Altitud"].max(), 100)
        freq_promedio = df["Frecuencia Respiratoria"].mean()
        predicciones = [model.predict([[alt, freq_promedio]])[0][0] for alt in altitud_range]
        ax1.plot(altitud_range, predicciones, color="red", label="Tendencia", linewidth=2)
        if altitud is not None and spo2 is not None:
            ax1.scatter(altitud, spo2, color="green", s=100, label="Predicción actual")
        ax1.set_xlabel("Altitud (metros)")
        ax1.set_ylabel("Saturación de Oxígeno (%)")
        ax1.set_title("Altitud vs Saturación de Oxígeno en Sangre")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Frecuencia vs SpO₂
        ax2.scatter(df["Frecuencia Respiratoria"], df["Saturacion Oxigeno (%)"], color="blue", label="Datos reales", alpha=0.6)
        frecuencia_range = np.linspace(df["Frecuencia Respiratoria"].min(), df["Frecuencia Respiratoria"].max(), 100)
        altitud_promedio = df["Altitud"].mean()
        predicciones_frec = [model.predict([[altitud_promedio, freq]])[0][0] for freq in frecuencia_range]
        ax2.plot(frecuencia_range, predicciones_frec, color="red", label="Tendencia", linewidth=2)
        if frecuencia is not None and spo2 is not None:
            ax2.scatter(frecuencia, spo2, color="green", s=100, label="Predicción actual")
        ax2.set_xlabel("Frecuencia Respiratoria (resp/min)")
        ax2.set_ylabel("Saturación de Oxígeno (%)")
        ax2.set_title("Frecuencia vs Saturación de Oxígeno en Sangre")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("static/images/regresion.png")
        plt.close()

    except Exception as e:
        print(f"Error al generar el gráfico: {e}")
        plt.figure(figsize=(8, 6))
        plt.scatter([1, 2, 3], [1, 2, 3])
        plt.title("Gráfico de respaldo")
        plt.savefig("static/images/regresion.png")
        plt.close()