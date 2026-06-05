import mlflow.pyfunc
import pandas as pd
import os
import sys
import mlflow
from mlflow import MlflowClient

print("MLflow OK", mlflow.__version__)

print("PYTHON:", sys.executable)
print("MLFLOW:", __import__("mlflow").__file__)

# Динамически вычисляем абсолютный путь к локальной БД
db_path = "/Users/dinislamalipkachev/PycharmProjects/MLOps/mlops-practice/mlflow.db"
tracking_uri = f"sqlite:///{db_path}"
mlflow.set_tracking_uri(tracking_uri)
sqlite_uri = f"sqlite:///{db_path}"

# Читаем URI DagsHub (если есть), иначе используем локальную абсолютную БД
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", sqlite_uri)
mlflow.set_tracking_uri(tracking_uri)

print(f"Подключаемся к реестру: {tracking_uri}")

# Используем синтаксис MLflow 3.x (Алиасы вместо Стадий)
model_name = "Iris_RF_Model"
alias = "champion"  # Заменили "Production" на алиас "champion"
print(os.path.abspath(db_path))
# Новый синтаксис: @ вместо /
model_uri = "models:/Iris_RF_Model/1"
print(f"Скачиваем модель по URI: {model_uri}")


client = MlflowClient()

print("\nTracking URI:", mlflow.get_tracking_uri())
print("\nRegistered models:")

for m in client.search_registered_models():
    print("-", m.name)

# Скачиваем боевую модель из реестра по её Алиасу (без путей к файлу!)
model = mlflow.pyfunc.load_model(
    "/Users/dinislamalipkachev/PycharmProjects/MLOps/mlops-practice/mlruns/1/models/m-cf2b9606a46c4fe08b12872e30d33f4c/artifacts"
)
# Тестовые данные (один цветок Ириса - 4 признака)
dummy_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
)

prediction = model.predict(dummy_data)
print(f"\n✅ Предсказание боевой модели: Класс {prediction[0]}")
