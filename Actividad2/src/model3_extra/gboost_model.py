import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GradientBoostingVGSales:
    
    def __init__(self):
        
        self.model = None
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        # Parámetros para el Gradient Boosting
        
        self.params = {'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 3, 'subsample': 0.9, 'random_state': 42}
        
    def load_data(self, filepath='../../data/processed/vgsales_limpio.csv'):
        
        print("Cargando dataset vgsales_limpio.csv...")
        
        try:
            self.df = pd.read_csv(filepath)
            
            print(f"Dataset cargado: {self.df.shape}")
            print(f"Registros: {len(self.df):,}")
            print(f"Columnas: {len(self.df.columns)}")
            return True
        
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {filepath}")
            return False
    
    def preprocess_data(self):
        print("\nPreprocesando datos...")
        
        df_clean = self.df.copy()
        # Eliminar Name y Rank (Rank causa data leakage)
        df_clean = df_clean.drop(['Name', 'Rank'], axis=1, errors='ignore')
        
        print("Valores faltantes antes:")
        
        missing_before = df_clean.isnull().sum()
        
        for column, missing in missing_before.items():
            if missing > 0:
                print(f"{column} : {missing}")
        
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce')
        df_clean['Year'].fillna(df_clean['Year'].median(), inplace=True)
        df_clean['Publisher'].fillna('Unknown', inplace=True)
        
        print("Aplicando One-Hot encoding...")
        
        df_clean = pd.get_dummies(df_clean, columns=['Platform', 'Genre', 'Publisher'], drop_first=True)
        
        self.X = df_clean.drop('Global_Sales', axis=1)
        self.y = df_clean['Global_Sales']
        self.feature_names = self.X.columns.tolist()
        
        print(f"Preprocesamiento completado | X: {self.X.shape}  y: {self.y.shape}")
        print(f"Features totales: {len(self.feature_names)}")
        print(f"Sin valores faltantes: {not self.X.isnull().any().any()}")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        
        print(f"\nDividiendo datos (test_size={test_size})...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        
        print(f"Train: {self.X_train.shape[0]:,} registros")
        print(f"Test:  {self.X_test.shape[0]:,} registros")
        
        return self
    
    def train_model(self, optimize=True):
        print("\nComenzando Entrenamiento Gradient Boosting Regressor...")
        
        if optimize:
            
            print("Optimizando hiperparámetros con GridSearch...")
            
            param_grid = {'n_estimators': [300, 400, 500], 'learning_rate': [0.03, 0.05, 0.07], 'max_depth': [3, 4], 'subsample': [0.8, 0.9], 'min_samples_split': [2, 5]}
            
            gb_base = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(gb_base, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
            
            grid_search.fit(self.X_train, self.y_train)
            self.model = grid_search.best_estimator_
            
            print(f"Mejores parámetros encontrados:")
            for parametro, valor in grid_search.best_params_.items():
                print(f"     {parametro}: {valor}")
                
            print(f"Mejor score CV: {grid_search.best_score_:.4f}")
            
        else:
            print("Usando parámetros optimizados por defecto...")
            
            self.model = GradientBoostingRegressor(**self.params)
            self.model.fit(self.X_train, self.y_train)
        
        self.is_trained = True
        print("Ahora su modelo está entrenado :)")
        
        return self
    
    def evaluate_model(self):
        
        if not self.is_trained or self.model is None:
            print("El modelo aún no ha sido entrenado")
            return None
        
        print("\nEvaluando modelo...")
        
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        
        train_r2 = r2_score(self.y_train, self.y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        
        test_r2 = r2_score(self.y_test, self.y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        
        print("\nResultados del Gradient Boosting:")
        print("=" * 50)
        print(f"{'Métrica':<15} {'Train':<12} {'Test':<12}")
        print("-" * 40)
        print(f"{'R²':<15} {train_r2:<12.4f} {test_r2:<12.4f}")
        print(f"{'RMSE':<15} {train_rmse:<12.4f} {test_rmse:<12.4f}")
        print(f"{'MAE':<15} {train_mae:<12.4f} {test_mae:<12.4f}")
        
        # Guardar metricas
        self.metrics = {'train': {'r2': train_r2, 'rmse': train_rmse, 'mae': train_mae}, 'test': {'r2': test_r2, 'rmse': test_rmse, 'mae': test_mae}}
        
        return self.metrics
    
    def analyze_features(self):
        
        if not self.is_trained:
            print("El modelo aún no ha sido entrenado")
            return None
        
        print("\nAnálisis de Feature Importance...")
        
        # Verificar que el modelo está entrenado
        
        if self.model is None:
            print("El modelo aún no ha sido entrenado")
            return None
        
        # Obtener importancias
        
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({'feature': self.feature_names, 'importance': importances, 'importance_pct': (importances / importances.sum()) * 100}).sort_values('importance', ascending=False)
        
        print("\nTop 10 Features más importantes:")
        print("-" * 50)
        for i, fila in feature_importance.head(10).iterrows():
            print(f"{fila['feature']:<15}: {fila['importance']:.4f} ({fila['importance_pct']:.1f}%)")
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def create_visualizations(self):
        
        if not self.is_trained:
            print("El modelo aún no ha sido entrenado")
            return
        
        print("\nGenerando visualizaciones...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis del Modelo Gradient Boosting', fontsize=16, fontweight='bold')
        
        axes[0,0].scatter(self.y_test, self.y_test_pred, alpha=0.6, color='steelblue')
        axes[0,0].plot([0, self.y_test.max()], [0, self.y_test.max()], 'r--', linewidth=2)
        axes[0,0].set_xlabel('Valores Reales')
        axes[0,0].set_ylabel('Predicciones')
        axes[0,0].set_title('Predicciones vs Valores Reales (Test)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Añadir R² al gráfico
        r2_test = r2_score(self.y_test, self.y_test_pred)
        axes[0,0].text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=axes[0,0].transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        residuals = self.y_test - self.y_test_pred
        axes[0,1].hist(residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0,1].set_xlabel('Residuos (Real - Predicción)')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].set_title('Distribución de Residuos')
        axes[0,1].grid(True, alpha=0.3)
        
        top_features = self.feature_importance.head(8)
        axes[1,0].barh(range(len(top_features)), top_features['importance'], color='lightseagreen', alpha=0.8)
        axes[1,0].set_yticks(range(len(top_features)))
        axes[1,0].set_yticklabels(top_features['feature'])
        axes[1,0].set_xlabel('Importancia')
        axes[1,0].set_title('Top 8 Features - Importancia')
        axes[1,0].grid(True, alpha=0.3, axis='x')
        
        axes[1,0].invert_yaxis()
        
        axes[1,1].scatter(self.y_test_pred, residuals, alpha=0.6, color='orange')
        axes[1,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1,1].set_xlabel('Predicciones')
        axes[1,1].set_ylabel('Residuos')
        axes[1,1].set_title('Residuos vs Predicciones')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar en carpeta local reports
        os.makedirs("./reports", exist_ok=True)
        fig_path = os.path.join("./reports", "gboost_analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Gráficos guardados en: {fig_path}")
        
        # Comentado para evitar bucles infinitos
        # plt.show()
        
        print("Visualizaciones generadas y guardadas")
    
    def save_model(self, filename='gboost_vgsales_optimized.pkl'):
        if not self.is_trained:
            print("El modelo aún no ha sido entrenado")
            return
        
        # Crear directorios locales si no existen
        os.makedirs("./reports", exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        
        models_dir = "./models/"
        reports_dir = "./reports/"
        
        # Guardar modelo y metadatos
        
        model_data = {'model': self.model,'feature_names': self.feature_names, 'training_params': self.params, 'metrics': self.metrics if hasattr(self, 'metrics') else None, 'feature_importance': self.feature_importance if hasattr(self, 'feature_importance') else None}
        
        model_path = os.path.join(models_dir, filename)
        joblib.dump(model_data, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        if hasattr(self, 'metrics'):
            metrics_path = os.path.join(reports_dir, 'gboost_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            print(f"Métricas guardadas en: {metrics_path}")
        
        if hasattr(self, 'feature_importance'):
            features_path = os.path.join(reports_dir, 'gboost_feature_importance.csv')
            self.feature_importance.to_csv(features_path, index=False)
            print(f"Feature importance guardada en: {features_path}")
            
        return model_path
    
    def generate_summary(self):
        if not self.is_trained:
            print("El modelo aún no ha sido entrenado")
            return
        
        print("\n" + "="*60)
        print("Resumen - Gradient Boosting")
        print("="*60)
        
        print(f"\nDataset:")
        print(f"Total registros: {len(self.df):,}")
        print(f"Features utilizadas: {len(self.feature_names)}")
        print(f"Train/Test split: {len(self.X_train)}/{len(self.X_test)}")
        
        print(f"\nMétricas finales:")
        print(f"R² Score: {self.metrics['test']['r2']:.4f}")
        print(f"RMSE: {self.metrics['test']['rmse']:.4f}")
        print(f"MAE: {self.metrics['test']['mae']:.4f}")
        
        print(f"\nTop 3 Features más importantes:")
        for idx, row in enumerate(self.feature_importance.head(3).iterrows(), start=1):
            print(f"{idx}. {row[1]['feature']}: {row[1]['importance_pct']:.1f}%")
        
        print("="*60)


def main():
    print("Modelo Gradient Boosting")
    print("="*60)
    print("Predicción de 'Global_Sales' usando GradientBoostingRegressor")
    print("Dataset: vgsales.csv")
    print("="*60)
    
    # Inicializar modelo
    gb_model = GradientBoostingVGSales()
    
    try:
        # 1. Cargar datos
        if not gb_model.load_data():
            return None
        
        # 2. Preprocesar
        gb_model.preprocess_data()
        
        # 3. Dividir datos
        gb_model.split_data()
        
        # 4. Entrenar
        gb_model.train_model(optimize=False)
        
        # 5. Evaluar
        gb_model.evaluate_model()
        
        # 6. Analizar features
        gb_model.analyze_features()
        
        # 7. Visualizar
        gb_model.create_visualizations()
        
        # 8. Guardar modelo
        gb_model.save_model()
        
        # 9. Resumen final
        gb_model.generate_summary()
        
        print("\nÁnalisis realizado completamente")
        
        return gb_model
        
    except Exception as e:
        print(f"Ha ocurrido un error: {str(e)}")
        return None
    
if __name__ == "__main__":
    model = main()