import datetime as dt
import time
import logging

from zoneinfo import ZoneInfo
from pyspark.sql import SparkSession, DataFrame

from airflow import DAG
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.providers.apache.spark.hooks.spark_connect import SparkConnectHook

logger = logging.getLogger(__name__)

doc_md = """
    대상 테이블을 불러와서 Object Storage에 동기화하는 DAG입니다.
    컬럼 타입이 불확실한 경우, 문자열 타입으로 저장됩니다.

    Data Connection ID가 Airflow에 등록되어 있어야합니다.
    Spark Connect 타입의 Connection "SPARK"가 Airflow에 등록되어 있어야합니다. 

    1. query_or_table: 동기화할 쿼리 또는 테이블 이름
    2. data_connection_id: Airflow Connection ID (예: MDM_DB)
    3. output_table: 동기화된 테이블을 저장할 위치
"""

TIME_ZONE = Variable.get("TIME_ZONE")

with DAG(
    dag_id="spark_table_sync",
    schedule=None,
    start_date=dt.datetime(2024, 4, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    catchup=False,
    params={
        "query_or_table": Param("tobecore.zcode", type="string", description="Query or table name to sync"),
        "data_connection_id": Param("MDM_DB", type="string", description="Airflow Connection ID that holds the source table"),
        "output_table": Param("tbw_catalog.iceberg_test.tobecore_zcode", type="string", description="Output table name"),
    },
    doc_md=doc_md,
    max_active_runs=1,
) as dag:
    @task
    def sync_table(query_or_table: str, data_connection_id: str, output_table: str):
        start_time = time.time()
        logger.info(f'======================= process_splink_match start =======================')

        if not query_or_table or not data_connection_id:
            raise AirflowFailException("Both 'table_name' and 'data_connection_id' are required.")
        
        failed = False
        try:
            conn = BaseHook.get_connection(data_connection_id)
        except Exception as e:
            failed = True
        if not conn:
            failed = True

        if failed:
            end_time = time.time()
            logger.info(f'======================= process_splink_match fail {end_time - start_time} sec =======================')
            raise AirflowFailException(f"Failed to get connection for {data_connection_id}")

        spark_url = SparkConnectHook('SPARK').get_connection_url().split(";")[0]
        spark = SparkSession.builder.appName('spark_table_sync').remote(spark_url).getOrCreate()
 
        
        host = conn.host
        port = conn.port
        database = conn.schema
        user = conn.login
        password = conn.password
        dialect = conn.conn_type
        if dialect == 'generic':
            dialect = conn.description.strip()

        if dialect.startswith("postgres"):
            jdbc_url = f"jdbc:postgresql://{host}:{port}/{database}"
            driver = "org.postgresql.Driver"
        elif dialect.startswith("hana"):
            jdbc_url = f"jdbc:sap://{host}:{port}/"
            if database:
                jdbc_url += f"?databaseName={database}"
            driver = "com.sap.db.jdbc.Driver"
        elif dialect.startswith("mssql"):
            jdbc_url = f"jdbc:sqlserver://{host}:{port};DatabaseName={database};encrypt=true;trustServerCertificate=true;"
            driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
        elif dialect.startswith("oracle"):
            jdbc_url = f"jdbc:oracle:thin:@{host}:{port}:{database}"
            driver = "oracle.jdbc.driver.OracleDriver"
        else:
            raise AirflowFailException(f"Unsupported dialect: {dialect}")
        

        reader = (
            spark.read
            .format("jdbc")
            .option("url", jdbc_url)
            .option("dbtable", query_or_table)
            .option("user", user)
            .option("password", password)
            .option("driver", driver)
        )

        df: DataFrame = reader.load()

        df.printSchema()

        df.writeTo(output_table).createOrReplace()

        spark.stop()

        end_time = time.time()
        logger.info(f'======================= process_splink_match end {end_time - start_time} sec =======================')

    sync_table_task = sync_table(
        query_or_table="{{ params.query_or_table }}",
        data_connection_id="{{ params.data_connection_id }}",
        output_table="{{ params.output_table }}",
    )