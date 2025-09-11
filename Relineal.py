import matplotlib
matplotlib.use('Agg')  # ✅ Usa backend no interactivo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de ejemplo para altitud y cantidad de sangre
data = {
    "Altitud": [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 
                500, 1500, 2000, 3000, 1000, 2500, 3500, 4000, 0, 2000],
    "Frecuencia Respiratoria": [12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                13, 17, 19, 23, 15, 21, 25, 27, 11, 18],
    "Cantidad Sangre": [5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8,
                        5.1, 5.5, 5.7, 6.1, 5.3, 5.9, 6.3, 6.5, 4.9, 5.7]
}

df = pd.DataFrame(data)

# Definimos Variables
x = df[["Altitud", "Frecuencia Respiratoria"]]
y = df[["Cantidad Sangre"]]

model = LinearRegression()
model.fit(x, y)

# Predecimos cantidad de sangre
def CalculateBlood(altitud: float, frecuencia: float):
    """Predecimos la cantidad de sangre a partir de altitud y frecuencia respiratoria"""
    result = model.predict([[altitud, frecuencia]])[0][0]
    return round(float(result), 2)

# Graficar (para usar en Flask)
def save_plot(altitud=None, frecuencia=None, cantidad_sangre=None):
    try:
        # Cerrar todas las figuras previas para evitar conflictos
        plt.close('all')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico 1: Altitud vs Cantidad de Sangre
        ax1.scatter(df["Altitud"], df["Cantidad Sangre"], color="blue", label="Datos reales", alpha=0.6)
        
        # Tendencia para altitud
        altitud_range = np.linspace(df["Altitud"].min(), df["Altitud"].max(), 100)
        freq_promedio = df["Frecuencia Respiratoria"].mean()
        predicciones = [model.predict([[alt, freq_promedio]])[0][0] for alt in altitud_range]
        
        ax1.plot(altitud_range, predicciones, color="red", label="Tendencia", linewidth=2)
        
        if altitud is not None and cantidad_sangre is not None:
            ax1.scatter(altitud, cantidad_sangre, color="green", s=100, label="Predicción actual")
        
        ax1.set_xlabel("Altitud (metros)")
        ax1.set_ylabel("Cantidad de Sangre (litros)")
        ax1.set_title("Altitud vs Cantidad de Sangre")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Frecuencia vs Cantidad de Sangre
        ax2.scatter(df["Frecuencia Respiratoria"], df["Cantidad Sangre"], color="blue", label="Datos reales", alpha=0.6)
        
        # Tendencia para frecuencia
        frecuencia_range = np.linspace(df["Frecuencia Respiratoria"].min(), df["Frecuencia Respiratoria"].max(), 100)
        altitud_promedio = df["Altitud"].mean()
        predicciones_frec = [model.predict([[altitud_promedio, freq]])[0][0] for freq in frecuencia_range]
        
        ax2.plot(frecuencia_range, predicciones_frec, color="red", label="Tendencia", linewidth=2)
        
        if frecuencia is not None and cantidad_sangre is not None:
            ax2.scatter(frecuencia, cantidad_sangre, color="green", s=100, label="Predicción actual")
        
        ax2.set_xlabel("Frecuencia Respiratoria (resp/min)")
        ax2.set_ylabel("Cantidad de Sangre (litros)")
        ax2.set_title("Frecuencia vs Cantidad de Sangre")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("static/images/regresion.png")
        plt.close()
        
    except Exception as e:
        print(f"Error al generar el gráfico: {e}")
        # Crear un gráfico simple de respaldo
        plt.figure(figsize=(8, 6))
        plt.scatter([1, 2, 3], [1, 2, 3])
        plt.title("Gráfico de respaldo")
        plt.savefig("static/images/regresion.png")
        plt.close()