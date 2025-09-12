import pandas as pd

# Cargar el dataset
df = pd.read_csv('vgsales.csv')

print("=== ANTES DE LIMPIAR ===")
print(f"Filas totales: {df.shape[0]}")
print("Datos faltantes:")
print(df.isnull().sum())

# LIMPIAR DATOS FALTANTES

# 1. Para Year: poner el año más común
year_mas_comun = df['Year'].mode()[0]  # El año que más se repite
df['Year'].fillna(year_mas_comun, inplace=True)

# 2. Para Publisher: poner "Desconocido"  
df['Publisher'].fillna('Desconocido', inplace=True)

# 3. Eliminar filas con ventas raras (0 o muy altas)
df = df[df['Global_Sales'] > 0]  # Eliminar ventas = 0
df = df[df['Global_Sales'] < 100]  # Eliminar ventas súper altas

print("\n=== DESPUÉS DE LIMPIAR ===")
print(f"Filas totales: {df.shape[0]}")
print("Datos faltantes:")
print(df.isnull().sum())

# Guardar el dataset limpio
df.to_csv('vgsales_limpio.csv', index=False)
print("\n✅ Dataset limpio guardado como 'vgsales_limpio.csv'")