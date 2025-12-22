from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    "streaming_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="*/1 * * * *",  # toutes les minutes
    catchup=False,
) as dag:

    consume = BashOperator(
        task_id="consume_microbatch",
        bash_command="python /opt/airflow/streaming/consumer.py"
    )

    consume
