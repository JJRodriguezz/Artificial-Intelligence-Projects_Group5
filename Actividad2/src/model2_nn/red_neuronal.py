import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

print("=== RED NEURONAL PARA PREDICCIÓN DE VENTAS ===")

# 1. CARGAR DATOS PREPROCESADOS
print("\n1. Cargando datos preprocesados...")

X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

print(f"✅ Entrenamiento: {X_train.shape}")
print(f"✅ Prueba: {X_test.shape}")
print(f"✅ Variables de entrada: {X_train.shape[1]}")

# 2. CREAR LA ARQUITECTURA DE LA RED NEURONAL
print("\n2. Creando arquitectura de la red neuronal...")

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Una salida para regresión
])

# 3. COMPILAR EL MODELO
print("✅ Arquitectura creada:")
print("- Capa 1: 64 neuronas + ReLU + Dropout")
print("- Capa 2: 32 neuronas + ReLU + Dropout") 
print("- Capa 3: 16 neuronas + ReLU")
print("- Salida: 1 neurona (ventas globales)")

model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error para regresión
    metrics=['mae']  # Mean Absolute Error
)

print("\n✅ Modelo compilado (optimizador: Adam, pérdida: MSE)")

# 4. ENTRENAR LA RED NEURONAL
print("\n4. Entrenando la red neuronal...")
print("⏳ Esto puede tomar unos minutos...")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("✅ Entrenamiento completado!")

# 5. HACER PREDICCIONES
print("\n5. Evaluando el modelo...")

# Predicciones en conjunto de prueba
y_test_pred = model.predict(X_test).flatten()

# Predicciones en conjunto de entrenamiento
y_train_pred = model.predict(X_train).flatten()

# 6. CALCULAR MÉTRICAS COMPLETAS
# Métricas de entrenamiento
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

# Métricas de prueba
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

# 7. MOSTRAR RESULTADOS
print("\n=== RESULTADOS DE LA RED NEURONAL ===")

print("\n📊 MÉTRICAS DE ENTRENAMIENTO:")
print(f"   MSE: {train_mse:.4f}")
print(f"   MAE: {train_mae:.4f}")
print(f"   RMSE: {train_rmse:.4f}")
print(f"   R²: {train_r2:.4f}")

print("\n📊 MÉTRICAS DE PRUEBA:")
print(f"   MSE: {test_mse:.4f}")
print(f"   MAE: {test_mae:.4f}")
print(f"   RMSE: {test_rmse:.4f}")
print(f"   R²: {test_r2:.4f}")

# 8. ANÁLISIS DE OVERFITTING
print("\n🔍 ANÁLISIS DE OVERFITTING:")
diferencia_r2 = train_r2 - test_r2
if diferencia_r2 > 0.05:
    print("⚠️ Posible overfitting detectado")
else:
    print("✅ No hay señales significativas de overfitting")

print(f"   Diferencia R² (train - test): {diferencia_r2:.4f}")

# Interpretación
print(f"\n🎯 En promedio, el modelo se equivoca por {test_mae:.2f} millones en las ventas")
if test_r2 > 0.8:
    print("🎉 ¡Excelente precisión!")
elif test_r2 > 0.6:
    print("👍 Buena precisión")
else:
    print("⚠️ Precisión mejorable")

# 9. GUARDAR RESULTADOS
print("\n9. Guardando resultados...")

# Guardar métricas completas
metricas_completas = {
    'entrenamiento': {
        'MSE': train_mse,
        'MAE': train_mae,
        'RMSE': train_rmse,
        'R2': train_r2
    },
    'prueba': {
        'MSE': test_mse,
        'MAE': test_mae,
        'RMSE': test_rmse,
        'R2': test_r2
    },
    'overfitting': {
        'diferencia_R2': diferencia_r2,
        'hay_overfitting': diferencia_r2 > 0.05
    }
}

with open('metricas_red_neuronal.json', 'w') as f:
    json.dump(metricas_completas, f, indent=2)

# Guardar modelo
model.save('modelo_red_neuronal.keras')

print("✅ Resultados guardados:")
print("- metricas_red_neuronal.json")
print("- modelo_red_neuronal.keras")

print("\n=== RESUMEN FINAL ===")
print(f"Variable objetivo: Global_Sales (ventas globales en millones)")
print(f"Precisión del modelo: {test_r2:.1%}")
print(f"Error promedio: {test_mae:.3f} millones de copias")