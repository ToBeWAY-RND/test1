import time
import json
import datetime as dt
import logging

from tbwlib.aiml import (
    OutlierModel,
    EMBEDDING_MODELS,
    OUTLIER_MODELS,
    get_embedding_count
)

from tbwlib.mdm import (
    DOMAINIDS,
    CODEI_DOMAINID,
    get_dimension,
    delete_outlier,
    insert_outlier
)

from tbwlib.utils import database as db, pandas as pd
from typing import List
from zoneinfo import ZoneInfo


from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException, AirflowFailException

logger = logging.getLogger(__name__)

TIME_ZONE = Variable.get("TIME_ZONE")

TASK_SIZE = 3
MINIMUM_BATCH_SIZE = 10000

@task
def delete_current_outliers(embedding_model_id: str, outlier_type: str, domainid: str, classid: str) -> str:
    start_time = time.time()
    logger.info('======================= delete_current_outliers start =======================')
    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    outlier_type = outlier_type.upper()

    if embedding_model_id not in EMBEDDING_MODELS:
        raise AirflowFailException(f"Invalid embedding model id: {embedding_model_id}, valid models are: {EMBEDDING_MODELS}")
    
    if outlier_type not in OUTLIER_MODELS:
        raise AirflowFailException(f"Invalid outlier check type: {outlier_type}. Valid check types are: {OUTLIER_MODELS}")
    
    if domainid not in DOMAINIDS:
        raise AirflowFailException(f"Invalid domain id: {domainid}. Valid domain ids are: {DOMAINIDS}")
    
    if classid == domainid or (domainid == CODEI_DOMAINID and classid == 'ROOT'):
        classid = None

    logger.info("Deleting current outlier results")
    delete_outlier(domainid, classid, embedding_model_id, outlier_type)

    end_time = time.time()
    logger.info(f'======================= delete_current_outliers end {end_time - start_time} sec =======================')
    return embedding_model_id

@task
def get_classids_for_outlier(embedding_model_id: str, domainid: str, classid: str) -> list:
    start_time = time.time()
    logger.info('======================= get_classids_for_outlier start =======================')
    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None

    records = get_embedding_count(domainid, embedding_model_id, parent_classid=classid)

    if records.empty:
        end_time = time.time()
        logger.info(f'======================= get_classids_for_outlier end {end_time - start_time} sec =======================')
        raise AirflowSkipException(f"No records to request outlier for {domainid} {embedding_model_id} under {classid}")
    
    records['batchid'] = pd.partition_with_lpt(records, TASK_SIZE, MINIMUM_BATCH_SIZE)
    classids_list = records.groupby('batchid')['classid'].apply(list).tolist()

    end_time = time.time()
    logger.info(f'======================= get_classids_for_outlier end {end_time - start_time} sec =======================')
    return classids_list

@task
def process_outlier_detection(embedding_model_id: str, outlier_type: str, domainid: str, classids: List[str], max_match_count: str, min_sentence_length: str, extra_params: str):
    start_time = time.time()
    logger.info('======================= process_outlier_detection start =======================')
    
    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    outlier_type = outlier_type.upper()
    max_match_count = int(max_match_count) if max_match_count != 'None' and max_match_count else 0
    min_sentence_length = int(min_sentence_length) if min_sentence_length != 'None' and min_sentence_length else 0
    extra_params = json.loads(extra_params) if extra_params != 'None' and extra_params else {}

    dimension = get_dimension(embedding_model_id)
    if dimension is None:
        raise AirflowFailException(f"Cannot find dimension for {embedding_model_id}")

    outlier_model = OutlierModel.create(
        embedding_model_id=embedding_model_id,
        outlier_check=outlier_type,
        domainid=domainid,
        dimension=dimension,
        max_match_count=max_match_count,
        min_sentence_length=min_sentence_length,
        params=extra_params
    )

    if not outlier_model:
        raise AirflowFailException(f"Failed to create outlier model for {embedding_model_id}")
    
    logger.info(f"Processing outlier detection for {domainid} {embedding_model_id} under {classids}")
    records = outlier_model.find(classids)

    if records.empty:
        logger.info(f"No outlier records found for {domainid} {embedding_model_id} under {classids}")
        end_time = time.time()
        logger.info(f'======================= process_outlier_detection end {end_time - start_time} sec =======================')
        return
    
    records['creationdtime'] = pd.Timestamp.now(tz=TIME_ZONE).to_pydatetime()
    insert_outlier(embedding_model_id, outlier_type, domainid, records)

    end_time = time.time()
    logger.info(f'======================= process_outlier_detection end {end_time - start_time} sec =======================')

doc_md = """
    도메인 분류 별 아웃라이더를 찾아내기 위한 DAG입니다.
    Airflow의 Connection 접속정보가 사전에 정의되어 있어야 합니다.
    Prerequisite : Airflow Connection Info (tobeway_aiml, tobeway_mdm)
    Params {
        domainid : 대상 도메인,
        classid : 대상 시작 클래스 id, None 입력 시 도메인 전체 대상
        outlier_check : 검사 방법,
        embedding_model_id : 임베딩 모델 id,
        max_match_count : 매칭 결과 최대 개수,
        min_sentence_length : 최소 문장 길이,
        extra_params :  추가 파라미터 
    }
    1. delete_current_outliers : 현재 아웃라이더 결과 삭제
    2. get_classids_for_outlier : 아웃라이더 검사를 위한 클래스 id 목록 가져오기
    3. process_outlier_detection : 아웃라이더 검사 수행
"""

with DAG(
    dag_id="aiml_class_outlier_batch",
    schedule=None,
    start_date=dt.datetime(2024, 4, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    catchup=False,
    params={
        "domainid": Param('MATERIAL', type="string", enum=DOMAINIDS),
        "classid" : Param(None, type=["null", "string"]),
        "outlier_check": Param('IQR', type="string", enum=OUTLIER_MODELS),
        "embedding_model_id": Param('VCA1', type="string", enum=EMBEDDING_MODELS),
        "max_match_count": Param(3, type="integer"),
        "min_sentence_length": Param(0, type="integer"),
        "extra_params": Param(None, type=["object", "null"])
    },
    doc_md=doc_md,
    max_active_runs=1,
) as dag:
    embedding_model_id = delete_current_outliers(
        embedding_model_id="{{ params.embedding_model_id }}",
        outlier_type="{{ params.outlier_check }}",
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}"
    )
    
    classids_list = get_classids_for_outlier(
        embedding_model_id=embedding_model_id,
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}"
    )

    process_outlier_detection.partial(
        embedding_model_id=embedding_model_id,
        outlier_type="{{ params.outlier_check }}",
        domainid="{{ params.domainid }}",
        max_match_count="{{ params.max_match_count }}",
        min_sentence_length="{{ params.min_sentence_length }}",
        extra_params="{{ params.extra_params }}"
    ).expand(classids=classids_list)