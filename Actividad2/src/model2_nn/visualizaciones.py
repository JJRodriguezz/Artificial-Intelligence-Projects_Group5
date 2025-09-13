import numpy as np
import matplotlib.pyplot as plt
import json

print("=== GENERANDO VISUALIZACIONES DE LA RED NEURONAL ===")

# 1. CARGAR DATOS Y RECREAR PREDICCIONES
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Recrear el modelo y hacer predicciones
import tensorflow as tf
from tensorflow import keras

# Recrear modelo con la misma arquitectura
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_test.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Cargar pesos guardados
try:
    model.load_weights('modelo_red_neuronal.h5')
    print("✅ Pesos del modelo cargados correctamente")
except:
    print("⚠️ No se pudieron cargar los pesos, usando predicciones aproximadas")

# 2. HACER PREDICCIONES
print("\n1. Generando predicciones...")
y_pred = model.predict(X_test).flatten()

# 3. CARGAR MÉTRICAS
with open('metricas_red_neuronal.json', 'r') as f:
    metricas = json.load(f)

# 4. CREAR VISUALIZACIONES
plt.figure(figsize=(15, 10))

# Gráfico 1: Predicciones vs Valores Reales
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, s=20, color='blue')
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--', lw=2, label='Predicción perfecta')
plt.xlabel('Ventas Reales (millones)')
plt.ylabel('Ventas Predichas (millones)')
plt.title('Predicciones vs Valores Reales')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Distribución de errores
plt.subplot(2, 2, 2)
errores = y_test - y_pred
plt.hist(errores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Error (Real - Predicho)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Errores')
plt.axvline(x=0, color='red', linestyle='--', label='Error = 0')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 3: Métricas del modelo
plt.subplot(2, 2, 3)
metricas_nombres = ['MSE', 'MAE', 'R²']
valores = [metricas['MSE'], metricas['MAE'], metricas['R2']]
colores = ['red', 'orange', 'green']

bars = plt.bar(metricas_nombres, valores, color=colores, alpha=0.7)
plt.title('Métricas de Rendimiento')
plt.ylabel('Valor')

# Agregar valores encima de las barras
for bar, valor in zip(bars, valores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{valor:.4f}', ha='center', va='bottom')

# Gráfico 4: Análisis de residuos
plt.subplot(2, 2, 4)
plt.scatter(y_pred, errores, alpha=0.6, s=20)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos (Real - Predicho)')
plt.title('Análisis de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados_red_neuronal.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. ESTADÍSTICAS ADICIONALES
print("\n=== ANÁLISIS DETALLADO ===")
print(f"📊 MSE: {metricas['MSE']:.4f}")
print(f"📊 MAE: {metricas['MAE']:.4f}")
print(f"📊 R²: {metricas['R2']:.4f}")

print(f"\n🎯 Error promedio: {abs(errores.mean()):.4f} millones")
print(f"🎯 Desviación estándar del error: {errores.std():.4f}")
print(f"🎯 Error máximo: {abs(errores).max():.4f} millones")

# Análisis por rangos de ventas
ventas_bajas = y_test < 1
ventas_medias = (y_test >= 1) & (y_test < 5)
ventas_altas = y_test >= 5

print(f"\n📈 Precisión por rango de ventas:")
print(f"   Bajas (<1M): MAE = {abs(errores[ventas_bajas]).mean():.4f}")
print(f"   Medias (1-5M): MAE = {abs(errores[ventas_medias]).mean():.4f}")
print(f"   Altas (>5M): MAE = {abs(errores[ventas_altas]).mean():.4f}")

print("\n✅ Visualizaciones guardadas como 'resultados_red_neuronal.png'")