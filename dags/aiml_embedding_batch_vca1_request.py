import time
import logging
import openai
import io
import json
import datetime as dt

from tbwlib.utils import database as db, pandas as pd
from tbwlib.mdm import (
    DOMAINIDS,
    get_default_model_name,
    get_dimension,
    get_embedding_records
)

from embedding_tasks import cleanup_stale_embeddings

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowFailException, AirflowSkipException

from zoneinfo import ZoneInfo

EMBEDDING_MODEL_ID = 'VCA1'

MAX_BATCH_SIZE = 50000
MAX_INPUT_SIZE = 100

USER_ID = 'tobeway'

TIME_ZONE = Variable.get("TIME_ZONE")
OPENAI_KEY = Variable.get("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

@task
def request_embeddings(domainid: str, classid: str, embedding_model_name: str, is_force: str, req_usrid):
    start_time = time.time()
    logger.info('======================= request_embeddings start =======================')
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    is_force = is_force == 'True' or is_force == True
    embedding_model_name = embedding_model_name if embedding_model_name != 'None' and embedding_model_name else None
    
    if classid is None:
        classid = domainid
    dimension = get_dimension(EMBEDDING_MODEL_ID)

    if not embedding_model_name:
        embedding_model_name = get_default_model_name(EMBEDDING_MODEL_ID)
    
    error_req = pd.DataFrame([{
        'batchid': f'batch_req_{pd.get_unique_id()}'[:50],
        'modelid': EMBEDDING_MODEL_ID,
        'requestdtime': pd.Timestamp.now(tz=TIME_ZONE).to_pydatetime(),
        'no_total': 0,
        'req_usrid': req_usrid,
        'domainid': domainid,
        'classid': classid,
        'embedding_model_name': embedding_model_name,
        'dimension': dimension,
        'is_force': 'Y' if is_force else 'N'
    }])
    
    if not embedding_model_name or not dimension:
        error_req['status'] = 'FAILED'
        db.insert_df(error_req, 'batch_req')

        end_time = time.time()
        logger.info(f'======================= request_embeddings end {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Failed to get embedding model name or dimension for {domainid} {classid}.")
 
    status = ['N', 'Y'] if is_force else 'N'
    records = get_embedding_records(EMBEDDING_MODEL_ID, domainid, classid, status, include_class=True)
    records['sentence'] = records['sentence'].str.strip()
    records = records[records['sentence'].str.len() > 0]

    if records.empty:
        error_req['status'] = 'SKIPPED'
        db.insert_df(error_req, 'batch_req')

        end_time = time.time()
        logger.info(f'======================= request_embeddings end {end_time - start_time} sec =======================')
        raise AirflowSkipException(f"No records to process for {domainid} {classid}.")

    openai_client = openai.Client(api_key=OPENAI_KEY)

    mast_records = len(records[records['mastid'].notnull()])
    class_records = len(records) - mast_records

    logger.info(f"Requesting embeddings for {len(records)} sentences: {mast_records} mast records, {class_records} class records")
    records = records.sort_values(by=['classid', 'mastid']).reset_index(drop=True)

    records['row_num'] = records.index

    records['batch_temp_id'] = records['row_num'] // MAX_BATCH_SIZE
    records['row_num'] = records.groupby('batch_temp_id', group_keys=False).cumcount()

    records['customid'] = records['row_num'] // MAX_INPUT_SIZE

    def get_first_classid_mastid(group):
        group['customid'] = f'{group["classid"].iloc[0]}_{group["mastid"].iloc[0] or "CLASS_SENTENCE"}'
        return group

    records = records.groupby('customid', group_keys=False).apply(get_first_classid_mastid)
    records['seq'] = records.groupby(['batch_temp_id', 'customid'], group_keys=False).cumcount() + 1

    records['domainid'] = domainid

    request_df = (
        records.sort_values(by=['seq'])
            .groupby(['batch_temp_id', 'customid'], as_index=False)
            .agg({'sentence': list})
    )

    for batch_temp_id, group in request_df.groupby('batch_temp_id'):
        data = []
        for _, row in group.iterrows():
            body = {
                'input': row['sentence'],
                'model': embedding_model_name,
            }
            if embedding_model_name.startswith('text-embedding-3'):
                body['dimensions'] = dimension
            data.append({
                'custom_id': row['customid'],
                'method': 'POST',
                'url': '/v1/embeddings',
                'body': body
            })
        data_str = '\n'.join(json.dumps(d) for d in data)
        file = io.BytesIO(data_str.encode('utf-8'))
        batch_files = openai_client.files.create(file=file, purpose='batch')
        file.close()

        batches = openai_client.batches.create(
            completion_window='24h', 
            endpoint="/v1/embeddings", 
            input_file_id=batch_files.id
        )
        records.loc[records['batch_temp_id'] == batch_temp_id, 'batchid'] = batches.id
        records.loc[records['batch_temp_id'] == batch_temp_id, 'status'] = 'REQUESTED'
        records.loc[records['batch_temp_id'] == batch_temp_id, 'message'] = batches.to_json()

    records['modelid'] = EMBEDDING_MODEL_ID
    detail_df = records[['batchid', 'modelid', 'customid', 'seq', 'domainid', 'mastid', 'classid', 'sentence']]
    detail_df.loc[detail_df['mastid'] == '', 'mastid'] = None

    records = records.groupby(['batchid', 'modelid', 'status', 'message'], as_index=False).size().rename(columns={'size': 'no_total'})
    records['requestdtime'] = pd.Timestamp.now(tz=TIME_ZONE).to_pydatetime()
    records['req_usrid'] = req_usrid
    records['domainid'] = domainid
    records['classid'] = classid
    records['embedding_model_name'] = embedding_model_name
    records['dimension'] = dimension
    records['is_force'] = 'Y' if is_force else 'N'

    db.insert_df(records, 'batch_req')
    db.insert_df(detail_df, 'batch_req_detail')

    end_time = time.time()
    logger.info(f'======================= request_embeddings end {end_time - start_time} sec =======================')

doc_md = """
    도메인 별 OpenAI Batch API를 통한 Sentence Embedding 생성 요청
    Airflow의 Connection 접속정보가 사전에 정의되어 있어야 합니다.
    Prerequisite : Airflow Connection Info (tobeway_mdm, tobeway_aiml)
    Params {
        domainid : 대상 도메인,
        classid : 대상 시작 클래스,
        embedding_model_name : 임베딩 모델 이름,
        is_force : 강제 임베딩 생성 여부,
        req_usrid : 요청자 ID
    }
    1. cleanup_stale_embeddings : 삭제된 Mast에 대한 임베딩 삭제
    2. request_embeddings : OpenAI Batch API를 통한 임베딩 요청 
"""

with DAG(
    dag_id="aiml_embedding_batch_vca1_request",
    schedule=None,
    start_date=dt.datetime(2024, 4, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    catchup=False,
    params={
        "domainid": Param("MATERIAL", type="string", enum=DOMAINIDS),
        "classid": Param(None, type=["string", "null"]),
        "embedding_model_name" : Param(None, type=["null", "string"]),
        "is_force" : Param(False, type="boolean"),
        "req_usrid": Param(USER_ID, type="string")
    },
    doc_md=doc_md,
    max_active_runs=1,
) as dag:
    cleanup_stale_embeddings(
        embedding_model_id=EMBEDDING_MODEL_ID,
        domainid="{{ params.domainid }}",
        allowed_models=[EMBEDDING_MODEL_ID],
        allowed_domainids=DOMAINIDS,
    )
    request_embeddings(
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}",
        embedding_model_name="{{ params.embedding_model_name }}",
        is_force="{{ params.is_force }}",
        req_usrid="{{ params.req_usrid }}"
    )