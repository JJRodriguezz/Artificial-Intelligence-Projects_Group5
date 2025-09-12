# 2. Preprocesamiento de Datos

## 2.1 Análisis Inicial del Dataset

El dataset original de videojuegos (`vgsales.csv`) contenía **16,598 registros** con **11 variables**:
- Variables categóricas: Name, Platform, Genre, Publisher
- Variables numéricas: Rank, Year, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales

### Problemas Identificados:
- **271 valores faltantes** en la columna `Year` (1.63% del dataset)
- **58 valores faltantes** en la columna `Publisher` (0.35% del dataset)
- Presencia de registros con ventas globales igual a 0
- Algunos registros con ventas anormalmente altas que podrían ser outliers

## 2.2 Proceso de Limpieza Realizado

### a) Limpieza de Datos Faltantes

**Columna Year:**
- **Estrategia aplicada:** Imputación con el valor más frecuente (moda)
- **Código utilizado:**
```python
year_mas_comun = df['Year'].mode()[0]  # El año que más se repite
df['Year'].fillna(year_mas_comun, inplace=True)
```
- **Justificación:** Se utilizó la moda para mantener la distribución temporal natural de los lanzamientos de videojuegos
- **Resultado:** 271 valores faltantes fueron reemplazados

**Columna Publisher:**
- **Estrategia aplicada:** Imputación con valor categórico "Desconocido"
- **Código utilizado:**
```python
df['Publisher'].fillna('Desconocido', inplace=True)
```
- **Justificación:** Crear una nueva categoría permite conservar estos registros sin sesgar hacia ningún editor específico
- **Resultado:** 58 valores faltantes fueron reemplazados

### b) Filtrado de Registros Anómalos

**Ventas Globales:**
- **Código utilizado:**
```python
df = df[df['Global_Sales'] > 0]      # Eliminar ventas = 0
df = df[df['Global_Sales'] < 100]    # Eliminar ventas súper altas
```
- **Eliminados:** Registros con `Global_Sales = 0` (juegos sin ventas registradas)
- **Eliminados:** Registros con `Global_Sales > 100` (valores anormalmente altos, posibles errores)
- **Justificación:** Estos valores podrían introducir ruido en el modelo y afectar el rendimiento

### c) Código Completo de Limpieza

```python
import pandas as pd

# Cargar el dataset
df = pd.read_csv('vgsales.csv')

# 1. Imputar años faltantes con la moda
year_mas_comun = df['Year'].mode()[0]
df['Year'].fillna(year_mas_comun, inplace=True)

# 2. Imputar editores faltantes
df['Publisher'].fillna('Desconocido', inplace=True)

# 3. Filtrar registros anómalos
df = df[df['Global_Sales'] > 0]      # Eliminar ventas = 0
df = df[df['Global_Sales'] < 100]    # Eliminar outliers extremos

# 4. Guardar dataset limpio
df.to_csv('vgsales_limpio.csv', index=False)
```

## 2.3 Resultado Final

- **Dataset original:** 16,598 registros
- **Dataset limpio:** [Número final de registros después de la limpieza]
- **Datos faltantes:** 0 en todas las columnas
- **Calidad de datos:** Mejorada significativamente para el entrenamiento del modelo

El dataset limpio fue guardado como `vgsales_limpio.csv` y utilizado para el entrenamiento de todos los modelos de machine learning.