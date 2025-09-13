import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Cargar dataset limpio (ajusta la ruta)
df = pd.read_csv('../../data/processed/vgsales_limpio.csv')

print("=== PREPROCESAMIENTO PARA RED NEURONAL ===")
print(f"Dataset inicial: {df.shape}")

# 1. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
print("\n1. Codificando variables categóricas...")

# Crear codificadores para cada variable categórica
label_encoders = {}

# Codificar Platform
label_encoders['Platform'] = LabelEncoder()
df['Platform_encoded'] = label_encoders['Platform'].fit_transform(df['Platform'])

# Codificar Genre  
label_encoders['Genre'] = LabelEncoder()
df['Genre_encoded'] = label_encoders['Genre'].fit_transform(df['Genre'])

# Codificar Publisher
label_encoders['Publisher'] = LabelEncoder()
df['Publisher_encoded'] = label_encoders['Publisher'].fit_transform(df['Publisher'])

print(f"✅ Platform: {df['Platform'].nunique()} categorías → números 0-{df['Platform'].nunique()-1}")
print(f"✅ Genre: {df['Genre'].nunique()} categorías → números 0-{df['Genre'].nunique()-1}")
print(f"✅ Publisher: {df['Publisher'].nunique()} categorías → números 0-{df['Publisher'].nunique()-1}")

# Ver resultado
print("\nEjemplo de codificación:")
print(df[['Platform', 'Platform_encoded', 'Genre', 'Genre_encoded']].head())

# 2. SELECCIONAR VARIABLES PARA EL MODELO
print("\n2. Preparando datos para la red neuronal...")

# Variables de entrada (features)
features = ['Platform_encoded', 'Genre_encoded', 'Publisher_encoded', 'Year', 
           'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Variable objetivo (target)
target = 'Global_Sales'

X = df[features]
y = df[target]

print(f"Variables de entrada: {X.shape[1]} columnas")
print(f"Variable objetivo: {target}")

# 3. NORMALIZACIÓN (súper importante para redes neuronales)
print("\n3. Normalizando datos...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Datos normalizados (media=0, desviación=1)")

# 4. DIVISIÓN TRAIN/TEST
print("\n4. Dividiendo en entrenamiento y prueba...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"✅ Entrenamiento: {X_train.shape[0]} muestras")
print(f"✅ Prueba: {X_test.shape[0]} muestras")

# 5. GUARDAR DATOS PREPROCESADOS
import numpy as np

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\n✅ Datos guardados para entrenar la red neuronal:")
print("- X_train.npy, X_test.npy, y_train.npy, y_test.npy")