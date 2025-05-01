import time
import datetime as dt
import logging

from tbwlib.aiml import (
    EmbeddingModel,
    EMBEDDING_MODELS,
    update_embeddings
)
from tbwlib.mdm import (
    DOMAINIDS,
    get_embedding_records,
    get_default_model_name,
    get_dimension
)
from tbwlib.utils import database as db, pandas as pd
from embedding_tasks import cleanup_stale_embeddings

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException, AirflowFailException

from zoneinfo import ZoneInfo

OP_TYPE = 'EMBEDDING'

TASK_SIZE = 3
MINIMUM_BATCH_SIZE = 10000
EXEC_BATCH_SIZE = 1000
TIME_ZONE = Variable.get("TIME_ZONE")

logger = logging.getLogger(__name__)

@task
def get_records_for_embedding(domainid: str, classid: str, embedding_model_id: str, is_force: str):
    start_time = time.time()
    logger.info('======================= get_records_for_embedding start =======================')
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    embedding_model_id = embedding_model_id.upper()
    is_force = is_force == 'True' or is_force == True
    
    logger.info("Flushing embedding_queue for EMBEDDING")
    db.execute("""
        delete from embedding_queue
        where op_type = :op_type
    """, params={'op_type': OP_TYPE})

    logger.info("Fetching records for embedding")
    statuses = ['Y', 'N'] if is_force else 'N'
    
    records = get_embedding_records(embedding_model_id, domainid, classid, statuses, include_class=True)
    records['sentence'] = records['sentence'].str.strip()
    records = records[records['sentence'] != '']

    if records.empty:
        end_time = time.time()
        logger.info(f'======================= get_records_for_embedding end {end_time - start_time} sec =======================')
        raise AirflowSkipException(f"No records to request embeddings for {domainid} {embedding_model_id} under {classid}")

    total_count = len(records)
    class_count = len(records[records['mastid'].isna()])
    mast_count = total_count - class_count

    logger.info(f"Total records: {total_count} ({mast_count} mast records, {class_count} class records)")

    records['batchid'] = pd.partition_with_lpt(records, TASK_SIZE, MINIMUM_BATCH_SIZE)
    batchids = records['batchid'].unique().tolist()
    for batchid in batchids:
        records.loc[records['batchid'] == batchid, 'batchid'] = f"batch_{pd.get_unique_id()}"[:50]

    records = records[['batchid', 'modelid', 'domainid', 'classid', 'mastid', 'sentence']]
    records['op_type'] = OP_TYPE

    db.insert_df(records, 'embedding_queue')

    end_time = time.time()
    logger.info(f'======================= get_records_for_embedding end {end_time - start_time} sec =======================')
    return records['batchid'].unique().tolist()

@task
def generate_embeddings(domainid: str, embedding_model_id: str, embedding_model_name: str, batchid: str):
    start_time = time.time()
    logger.info('======================= generate_embeddings start =======================')
    domainid = domainid.upper()
    embedding_model_id = embedding_model_id.upper()
    embedding_model_name = embedding_model_name if embedding_model_name != 'None' and embedding_model_name else None

    logger.info(f"Reading records for embedding of batchid: {batchid}")

    if not embedding_model_name:
        embedding_model_name = get_default_model_name(embedding_model_id)
        if not embedding_model_name:
            end_time = time.time()
            logger.info(f'======================= generate_embeddings fail {end_time - start_time} sec =======================')
            raise AirflowFailException(f"Failed to get default model name for {embedding_model_id}.")
    
    dimension = get_dimension(embedding_model_id)
    if not dimension:
        end_time = time.time()
        logger.info(f'======================= generate_embeddings fail {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Failed to get dimension for {embedding_model_id}.")
    
    query = f"""
        select classid, mastid, sentence
        from embedding_queue
        where batchid = :batchid and op_type = :op_type
    """
    records = db.execute(query, params={'batchid': batchid, 'op_type': OP_TYPE})

    db.execute("""
        delete from embedding_queue
        where batchid = :batchid and op_type = :op_type
    """, params={'batchid': batchid, 'op_type': OP_TYPE})
    
    if records.empty:
        end_time = time.time()
        logging.info(f'======================= generate_embeddings end {end_time - start_time} sec =======================')
        raise AirflowSkipException(f'No records for batch {batchid}')

    total_count = len(records)
    class_count = len(records[records['mastid'].isna()])
    mast_count = total_count - class_count

    logger.info(f"Generating embeddings for {total_count} ({mast_count} mast records, {class_count} class records)")

    embedding_model = EmbeddingModel.create(embedding_model_id, domainid)

    for start_idx in range(0, len(records), EXEC_BATCH_SIZE):
        logger.info(f"Processing records {start_idx} to {min(start_idx + EXEC_BATCH_SIZE, len(records))}")
        batch = records.iloc[start_idx:start_idx + EXEC_BATCH_SIZE].copy().reset_index(drop=True)
    
        sentences = batch['sentence'].tolist()
        batch['embedding'] = embedding_model.embed(sentences)
        
        failed_request = len(batch[batch['embedding'].isna()])
        if failed_request > 0:
            logger.warning(f"Embedding failed for {failed_request} records")
        batch = batch[batch['embedding'].notnull()].reset_index(drop=True)
        
        if batch.empty:
            continue

        batch['creationdtime'] = pd.Timestamp.now(tz=TIME_ZONE).to_pydatetime()

        update_embeddings(batch, domainid, embedding_model_id, dimension)
    
    embedding_model.close()

    db.execute("""
        delete from embedding_queue
        where batchid = :batchid and op_type = :op_type
    """, params={'batchid': batchid, 'op_type': OP_TYPE})

    end_time = time.time()
    logger.info(f'======================= generate_embeddings end {end_time - start_time} sec =======================')

doc_md = """
    도메인 별 임베딩을 하기 위한 Job입니다.
    Airflow의 Connection 접속정보가 사전에 정의되어 있어야 합니다.
    Prerequisite : Airflow Connection Info (tobeway_mdm)
    Params {
        domainid (str): 도메인 ID (ex: MATERIAL, CUSTOMER, ...)
        classid (str): 시작 클래스 ID (ex: MATERIAL, ...)
        embedding_model_id (str): 임베딩 모델 ID (ex: VCM1, VCA1, ...)
        embedding_model_name (str): 임베딩 모델 이름
        is_force (bool): 강제 실행 여부 (기본값: False)
    }
    1. cleanup_stale_embeddings: 임베딩 테이블에서 orphaned 레코드를 삭제합니다.
    2. get_records_for_embedding: 임베딩을 위한 레코드를 가져와서 배치 레코드를 생성합니다.
    3. generate_embeddings: 배치 레코드를 기반으로 임베딩을 생성합니다.
"""

with DAG(
    dag_id="aiml_embedding_batch",
    schedule=None,
    start_date=dt.datetime(2024, 4, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    catchup=False,
    params={
        "domainid": Param("MATERIAL", type="string", enum=DOMAINIDS),
        "classid": Param(None, type=["string", "null"]),
        "embedding_model_id": Param("VCM1", type="string", enum=EMBEDDING_MODELS),
        "embedding_model_name" : Param(None, type=["null", "string"]),
        "is_force" : Param(False, type="boolean")
    },
    doc_md=doc_md,
    max_active_runs=1,
) as dag:
    cleanup_stale_embeddings(
        embedding_model_id="{{ params.embedding_model_id }}",
        domainid="{{ params.domainid }}",
        allowed_models=EMBEDDING_MODELS,
        allowed_domainids=DOMAINIDS,
    )
    batchids = get_records_for_embedding(
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}",
        embedding_model_id="{{ params.embedding_model_id }}",
        is_force="{{ params.is_force }}",
    )
    generate_embeddings.partial(
        domainid="{{ params.domainid }}",
        embedding_model_id="{{ params.embedding_model_id }}",
        embedding_model_name="{{ params.embedding_model_name }}",
    ).expand(batchid=batchids)