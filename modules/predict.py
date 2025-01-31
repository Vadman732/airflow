import dill
import pandas as pd
import os
import json
import logging

# Устанавливаем путь к проекту
path = os.environ.get('PROJECT_PATH', '..')

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def predict():
    def load_model():
        model_path = os.path.join(path, 'data', 'models')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Директория с моделями не найдена: {model_path}")

        # Поиск .pkl файла в папке models
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl')]
        if not model_files:
            raise FileNotFoundError("Не найдено ни одной модели (.pkl) в директории models")

        model_file = model_files[0]  # Берем первый найденный .pkl файл
        model_full_path = os.path.join(model_path, model_file)
        logging.info(f"Загрузка модели из {model_full_path}")

        with open(model_full_path, 'rb') as f:
            model = dill.load(f)
        return model

    def make_predictions(model):
        test_data_path = os.path.join(path, "data", "test")
        results = []

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Директория с тестовыми данными не найдена: {test_data_path}")

        for file_name in os.listdir(test_data_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(test_data_path, file_name)

                with open(file_path, 'r') as file:
                    data = json.load(file)

                    if isinstance(data, dict):
                        data = [data]

                    try:
                        data_df = pd.DataFrame(data)
                    except Exception as e:
                        logging.error(f"Ошибка при преобразовании данных в DataFrame: {e}")
                        continue

                    try:
                        prediction = model.predict(data_df)[0]
                        results.append({"file_name": file_name, "prediction": prediction})
                    except Exception as e:
                        logging.error(f"Ошибка при предсказании для файла {file_name}: {e}")

        if results:
            return pd.DataFrame(results)
        else:
            raise ValueError("Нет данных для предсказаний")

    def save_predictions(predictions_df):
        predictions_path = os.path.join(path, 'data', 'predictions', 'predictions_with_files.csv')
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        predictions_df.to_csv(predictions_path, index=False)
        logging.info(f"Предсказания сохранены по пути {predictions_path}")

    model = load_model()
    predictions_df = make_predictions(model)
    save_predictions(predictions_df)


if __name__ == '__main__':
    predict()

logging.info(f"Текущий рабочий каталог: {os.getcwd()}, Абсолютный путь: {os.path.abspath('.')}")
