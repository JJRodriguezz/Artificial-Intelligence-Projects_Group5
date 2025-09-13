"""
Análisis de Ventas de Videojuegos usando Support Vector Machine (SVM)
Dataset: vgsales.csv de Kaggle (después de limpieza)
Problema: Regresión - Predicción de Global_Sales (ventas globales)

Notas:
- Kernel poly eliminado por bajo rendimiento y alto tiempo de entrenamiento.
- Mejor análisis de errores para valores pequeños.
- Configuraciones optimizadas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class VGSalesAnalyzer:
    def __init__(self, csv_file=None):
        if csv_file is None:
            self.csv_file = 'data/processed/vgsales_limpio.csv'
        else:
            self.csv_file = csv_file
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.encoders = {}
        
    def load_data(self):
        """Cargar el dataset de Kaggle desde CSV"""
        try:
            print("Cargando dataset de Kaggle (vgsales.csv)...")
            self.data = pd.read_csv(self.csv_file)
            print(f"Dataset cargado exitosamente: {len(self.data)} filas, {len(self.data.columns)} columnas")
            print(f"Columnas disponibles: {list(self.data.columns)}")
            print(f"Fuente: Dataset de ventas de videojuegos de Kaggle")
            return True
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.csv_file}")
            return False
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return False
    
    def prepare_features(self):
        """Preparar características para el modelo SVM"""
        print("\nPreparando datos para análisis de regresión...")
        
        df = self.data.copy()
        target_column = 'Global_Sales'
        
        # Estadísticas de la variable objetivo
        print(f"Variable objetivo: {target_column} (ventas globales en millones)")
        print(f"Estadísticas de {target_column}:")
        print(f"  - Promedio: {df[target_column].mean():.2f} millones")
        print(f"  - Mediana: {df[target_column].median():.2f} millones")
        print(f"  - Rango: {df[target_column].min():.2f} - {df[target_column].max():.2f} millones")
        print(f"  - Desviación estándar: {df[target_column].std():.2f}")
        
        # Características principales (ventas regionales)
        regional_sales = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        feature_columns = []
        
        print("\nCaracterísticas utilizadas:")
        for col in regional_sales:
            if col in df.columns:
                feature_columns.append(col)
                print(f"  {col}: Correlación con Global_Sales = {df[col].corr(df[target_column]):.3f}")
        
        # Características numéricas adicionales
        numeric_features = ['Year']
        
        if 'Critic_Score' in df.columns:
            df['Critic_Score'] = df['Critic_Score'].fillna(df['Critic_Score'].median())
            feature_columns.append('Critic_Score')
            print(f"  Critic_Score agregado")
        
        if 'User_Score' in df.columns:
            df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
            df['User_Score'] = df['User_Score'].fillna(df['User_Score'].median())
            feature_columns.append('User_Score')
            print(f"  User_Score agregado y limpiado")
        
        for col in numeric_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Codificación de variables categóricas
        categorical_features = ['Platform', 'Genre']
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                feature_columns.append(f'{col}_encoded')
                print(f"  {col}: {len(le.classes_)} categorías codificadas")
        
        # Verificar características válidas
        if len(feature_columns) == 0:
            print("Error: No se encontraron características válidas")
            return False
        
        # Preparar matrices X e y
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remover filas con valores nulos
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        print(f"\nResumen de características:")
        print(f"Características finales: {len(feature_columns)}")
        print(f"Variable objetivo: {target_column}")
        print(f"Muestras válidas: {len(X)}")
        
        # División train-test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalización
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Division de datos:")
        print(f"  - Entrenamiento: {len(self.X_train)} muestras (80%)")
        print(f"  - Prueba: {len(self.X_test)} muestras (20%)")
        print("Preparacion completada")
        
        return True
    
    def train_svm_model(self, kernel='rbf', C=1.0, gamma='scale', epsilon=0.1):
        """Entrenar modelo SVM"""
        print(f"\nEntrenando SVM - Kernel: {kernel}, C: {C}, epsilon: {epsilon}")
        
        self.model = SVR(kernel=kernel, C=C, gamma=gamma, epsilon=epsilon)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        print("Modelo SVM entrenado exitosamente")
        return True
    
    def evaluate_model(self):
        """Evaluar el rendimiento del modelo SVM"""
        if self.model is None:
            print("Error: Primero debes entrenar el modelo")
            return False
            
        print("\nEvaluando modelo SVM...")
        
        # Predicciones
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Métricas
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Mostrar resultados
        print("=" * 70)
        print("RESULTADOS DEL MODELO SVM - REGRESION")
        print("=" * 70)
        print(f"Tecnica: Support Vector Machine (SVM)")
        print(f"Problema: Regresion - Prediccion de Global_Sales")
        print(f"Configuracion del modelo:")
        print(f"  - Kernel: {self.model.kernel.upper()}")
        print(f"  - Parametro C: {self.model.C}")
        print(f"  - Gamma: {self.model.gamma}")
        print(f"  - Epsilon: {self.model.epsilon}")
        print(f"  - Vectores de soporte: {len(self.model.support_)}")
        
        print("\nMETRICAS DE REGRESION:")
        print("-" * 65)
        print(f"{'Metrica':<15} {'Entrenamiento':<20} {'Prueba':<20}")
        print("-" * 65)
        print(f"{'MSE':<15} {train_mse:<20.4f} {test_mse:<20.4f}")
        print(f"{'RMSE':<15} {train_rmse:<20.4f} {test_rmse:<20.4f}")
        print(f"{'MAE':<15} {train_mae:<20.4f} {test_mae:<20.4f}")
        print(f"{'R2 Score':<15} {train_r2:<20.4f} {test_r2:<20.4f}")
        print("-" * 65)
        
        # Explicacion del rendimiento
        print(f"\nEXPLICACION DEL RENDIMIENTO:")
        if test_r2 >= 0.9:
            performance = "Excelente"
            explanation = "El modelo explica mas del 90% de la varianza"
        elif test_r2 >= 0.7:
            performance = "Muy bueno"
            explanation = "El modelo explica mas del 70% de la varianza"
        elif test_r2 >= 0.5:
            performance = "Bueno"
            explanation = "El modelo explica mas del 50% de la varianza"
        else:
            performance = "Necesita mejora"
            explanation = "El modelo necesita mejores caracteristicas o ajustes"
            
        print(f"Calidad del modelo: {performance}")
        print(f"{explanation}")
        print(f"Varianza explicada: {max(0, test_r2 * 100):.1f}%")
        
        # Analisis de errores para valores pequeños
        print(f"\nANALISIS DE ERRORES PARA VALORES PEQUEÑOS:")
        print("Los errores porcentuales altos se deben a valores reales muy pequeños")
        print("donde pequenos errores absolutos se convierten en grandes porcentajes")
        
        # Muestra de predicciones mejorada
        print("\nMUESTRA DE PREDICCIONES (con analisis):")
        print("-" * 75)
        print(f"{'Real':<8} {'Pred':<8} {'Error':<8} {'Error%':<12} {'Tipo':<15}")
        print("-" * 75)
        
        small_values = 0
        large_errors = 0
        
        for i in range(min(15, len(self.y_test))):
            real = self.y_test.iloc[i]
            pred = y_test_pred[i]
            error_abs = abs(real - pred)
            
            # Calcular error porcentual de forma segura
            if real > 0.1:  # Valores no tan pequeños
                error_pct = (error_abs / real) * 100
                value_type = "Normal"
            else:
                error_pct = (error_abs / max(real, 0.01)) * 100
                value_type = "Muy pequeño"
                small_values += 1
                if error_pct > 100:
                    large_errors += 1
            
            print(f"{real:<8.2f} {pred:<8.2f} {error_abs:<8.2f} {error_pct:<12.1f}% {value_type:<15}")
        
        print("-" * 75)
        print(f"Valores muy pequeños (<0.1M): {small_values}/{min(15, len(self.y_test))}")
        print(f"Errores >100% en pequeños: {large_errors}")
        
        # Guardar metricas
        self.final_metrics = {
            'technique': 'Support Vector Machine (SVM)',
            'problem_type': 'Regresion',
            'target_variable': 'Global_Sales',
            'kernel': self.model.kernel,
            'C': self.model.C,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'support_vectors': len(self.model.support_)
        }
        
        return test_r2
    
    def run_comparative_analysis(self):
        """Ejecutar análisis comparativo"""
        print("ANALISIS SVM - VENTAS DE VIDEOJUEGOS")
        print("=" * 60)
        print("Requerimientos:")
        print("- Dataset de Kaggle en formato CSV")
        print("- Support Vector Machine (SVM)")
        print("- Problema de regresion: Prediccion de Global_Sales")
        print("=" * 60)
        
        # Cargar y preparar datos
        if not self.load_data():
            return False
        
        if not self.prepare_features():
            return False
        
        # Configuraciones SVM (sin poly por bajo rendimiento)
        svm_configurations = [
            {'name': 'SVM Linear', 'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1},
            {'name': 'SVM RBF (C=1)', 'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
            {'name': 'SVM RBF (C=10)', 'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.1},
            {'name': 'SVM RBF (C=100)', 'kernel': 'rbf', 'C': 100.0, 'epsilon': 0.01}
        ]
        
        best_score = -np.inf
        best_config = None
        results_comparison = []
        
        print(f"\nANALISIS COMPARATIVO - CONFIGURACIONES:")
        print("-" * 60)
        
        for i, config in enumerate(svm_configurations, 1):
            try:
                print(f"\nConfiguracion {i}: {config['name']}")
                
                # Entrenar
                self.train_svm_model(
                    kernel=config['kernel'], 
                    C=config['C'],
                    epsilon=config['epsilon']
                )
                
                # Evaluacion
                y_pred = self.model.predict(self.X_test_scaled)
                r2 = r2_score(self.y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                mae = mean_absolute_error(self.y_test, y_pred)
                
                print(f"  R2 Score: {r2:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")
                
                results_comparison.append({
                    'name': config['name'],
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'config': config
                })
                
                if r2 > best_score:
                    best_score = r2
                    best_config = config
                    
            except Exception as e:
                print(f"  Error: {e}")
        
        # Tabla comparativa
        print(f"\nTABLA COMPARATIVA:")
        print("=" * 70)
        print(f"{'Configuracion':<20} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'Ranking':<10}")
        print("=" * 70)
        
        results_comparison.sort(key=lambda x: x['r2'], reverse=True)
        
        for i, result in enumerate(results_comparison, 1):
            ranking = "MEJOR" if i == 1 else f"#{i}"
            print(f"{result['name']:<20} {result['r2']:<10.4f} {result['rmse']:<10.4f} {result['mae']:<10.4f} {ranking:<10}")
        
        print("=" * 70)
        
        # Modelo final
        print(f"\nMODELO FINAL CON MEJOR CONFIGURACION:")
        print(f"  - Ganador: {best_config['name']}")
        print(f"  - R2 Score: {best_score:.4f}")
        
        self.train_svm_model(
            kernel=best_config['kernel'], 
            C=best_config['C'],
            epsilon=best_config['epsilon']
        )
        
        # Evaluacion detallada
        self.evaluate_model()
        
        print(f"\nANALISIS COMPLETADO")
        print(f"Variable objetivo: Global_Sales")
        
        return True

def main():
    """Funcion principal"""
    print("ANALISIS SVM - VENTAS DE VIDEOJUEGOS")
    print("Variable objetivo: Global_Sales")
    print()
    
    analyzer = VGSalesAnalyzer()
    analyzer.run_comparative_analysis()

if __name__ == "__main__":
    main()