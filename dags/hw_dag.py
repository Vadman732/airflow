import os
import datetime as dt
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator

# Устанавливаем путь к проекту
path = '/opt/airflow'
os.environ['PROJECT_PATH'] = path
sys.path.insert(0, path)  # Добавляем путь в sys.path до импорта модулей

# Импортируем функции из модулей
from modules.pipeline import pipeline
from modules.predict import predict

# Базовые аргументы DAG
default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}


# Определение DAG
with DAG(
        dag_id='model_pipeline',
        schedule=None,  # Новый вариант (вместо schedule_interval=None)
        default_args=default_args,
) as dag:

    # Задача: Выполнение пайплайна
    run_pipeline = PythonOperator(
        task_id='run_pipeline',
        python_callable=pipeline,
    )

    # Задача: Выполнение предсказаний
    run_predictions = PythonOperator(
        task_id='run_predictions',
        python_callable=predict,
    )

    # Порядок выполнения задач
    run_pipeline >> run_predictions

