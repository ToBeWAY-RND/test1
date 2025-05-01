import time
import datetime as dt
import json
import openai
import logging

from tbwlib.utils import database as db, pandas as pd
from tbwlib.aiml import update_embeddings
from tbwlib.mdm import DOMAINIDS

from airflow import DAG
from airflow.models.param import Param
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.decorators import task

from typing import List
from zoneinfo import ZoneInfo

API_STATUS_MAPPING = {
    "validating": "REQUESTED",
    "failed": "FAILED",
    "cancelled": "CANCELLED",
    "expired": "FAILED",
    "cancelling": "CANCELLED",
    "completed": "PROCESSING",
    "finalizing": "REQUESTED",
    "in_progress": "REQUESTED",
}

EMBEDDING_MODEL_ID = 'VCA1'
TASK_SIZE = 3
MINIMUN_BATCH_SIZE = 10000
EXEC_BATCH_SIZE = 30
TIME_ZONE = Variable.get("TIME_ZONE")
OPENAI_KEY = Variable.get("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

def delete_file_on_batch_request(openai_client: openai.Client, file_id: str):
    if file_id:
        try:
            openai_client.files.delete(file_id)
        except Exception as e:
            logging.warning(f"Error deleting file {file_id}: {e}. Please check manually.")

@task
def listen_to_responses(domainid: str):
    start_time = time.time()
    logger.info('======================= listen_to_responses start =======================')

    domainid = domainid.upper() if domainid.upper() != 'ALL' else None

    if domainid and domainid not in DOMAINIDS:
        end_time = time.time()
        logger.info(f'======================= listen_to_responses fail {end_time - start_time} sec =======================')
        raise AirflowSkipException(f"Invalid domainid: {domainid}. Available domainids: {DOMAINIDS}")

    query = "select batchid from batch_req where status = :status and modelid = :modelid"

    if domainid:
        query += f" and domainid = :domainid"
    
    params = {
        'domainid': domainid,
        'modelid': EMBEDDING_MODEL_ID,
        'status': 'REQUESTED'
    }
    batchids = db.execute(query, params=params)['batchid'].tolist()

    logger.info(f"Polling OpenAI API for batch status for {len(batchids)} batches")

    openai_client = openai.Client(api_key=OPENAI_KEY)

    batches = []

    for batchid in batchids:
        batch_status = openai_client.batches.retrieve(batchid)
        logger.info(f"Batch {batchid} status: {batch_status.status}")
        
        is_unknown_status = batch_status.status not in API_STATUS_MAPPING
        status = API_STATUS_MAPPING.get(batch_status.status, 'FAILED')

        batches.append({
            'batchid': batchid,
            'modelid': EMBEDDING_MODEL_ID,
            'status': status,
            'message': batch_status.to_json(),
        })

        if status in ['FAILED', 'CANCELLED']:
            logger.warning(f"Batch {batchid} failed or cancelled")
        
            if batch_status.errors:
                logger.warning(f"Errors: {batch_status.errors.to_json()}")
            
            if is_unknown_status:
                logger.warning(f"Batch status {batch_status.status} is unknown to the system")
                logger.warning("Please check the OpenAI Dashboard for more details.")
            else:
                logger.info(f"Deleting input / output / error files.")
                delete_file_on_batch_request(openai_client, batch_status.input_file_id)
                delete_file_on_batch_request(openai_client, batch_status.output_file_id)
                delete_file_on_batch_request(openai_client, batch_status.error_file_id)
            
            query = "delete from batch_req_detail where batchid = :batchid and modelid = :modelid"
            db.execute(query, params={'batchid': batchid, 'modelid': EMBEDDING_MODEL_ID})
    
    if batches:
        logger.info("Updating batch status in the database")
        batches = pd.DataFrame(batches, dtype='object')
        db.upsert_df(batches, 'batch_req', match_columns=['batchid', 'modelid'])

    query = "select batchid, no_total from batch_req where status = :status and modelid = :modelid"
    batches = db.execute(query, params={'status': 'PROCESSING', 'modelid': EMBEDDING_MODEL_ID})
    batches['no_total'] = batches['no_total'].astype(int)
    batches = batches[batches['no_total'] > 0].reset_index(drop=True)

    if batches.empty:
        end_time = time.time()
        logger.info(f'======================= listen_to_responses end {end_time - start_time} sec =======================')
        raise AirflowSkipException("No batches to process")
    
    logger.info(f"Found {len(batches)} batches to process")
    
    batches['chunkid'] = pd.partition_with_lpt(batches, TASK_SIZE, MINIMUN_BATCH_SIZE, weight_col='no_total')
    batchids = batches.groupby('chunkid')['batchid'].apply(list).tolist()
            
    end_time = time.time()
    logger.info(f'======================= listen_to_responses end {end_time - start_time} sec =======================')
    return batchids

@task
def process_batch_response(batchids: List[str]):
    start_time = time.time()
    logger.info('======================= process_batch_response start =======================')

    openai_client = openai.Client(api_key=OPENAI_KEY)

    for batchid in batchids:
        logger.info(f"Fetching batch status for {batchid}")
        batch_status = openai_client.batches.retrieve(batchid)
        file_response = openai_client.files.content(batch_status.output_file_id)
        json_list = file_response.text.strip().split('\n')

        query = "select domainid, dimension from batch_req where batchid = :batchid and modelid = :modelid"
        infos = db.execute(query, params={'batchid': batchid, 'modelid': EMBEDDING_MODEL_ID})
        if infos.empty:
            logger.warning(f"Batch {batchid} not found in the database")
            continue

        domainid = infos['domainid'].values[0]
        dimension = int(infos['dimension'].values[0])

        if not domainid or not dimension:
            logger.warning(f"Batch {batchid} has no domainid or dimension")
            continue

        for start_idx in range(0, len(json_list), EXEC_BATCH_SIZE):
            logger.info(f"Fetching batch {batchid} from {start_idx} to {min(start_idx + EXEC_BATCH_SIZE, len(json_list))}")
            records = []
            for json_str in json_list[start_idx:start_idx + EXEC_BATCH_SIZE]:
                if not json_str.strip():
                    continue
                file_content = json.loads(json_str)
                customid = file_content['custom_id']

                for data in file_content['response']['body']['data']:
                    embedding_val = data['embedding']
                    if isinstance(embedding_val, str):
                        embedding_val = json.loads(embedding_val)
                    records.append({
                        'batchid': batchid,
                        'modelid': EMBEDDING_MODEL_ID,
                        'domainid': domainid,
                        'customid': customid,
                        'seq': int(data['index']) + 1,
                        'embedding': embedding_val,
                    })

            if not records:
                logger.warning(f"No records found for batch {batchid} in range {start_idx} to {min(start_idx + EXEC_BATCH_SIZE, len(json_list))}")
                continue
            logger.info(f"Processing {len(records)} records for batch {batchid} in range {start_idx} to {min(start_idx + EXEC_BATCH_SIZE, len(json_list))}")
            records = pd.DataFrame(records, dtype='object')
            
            logger.info("Fetching metadata for chunk: domainid, classid, mastid, sentence")
            logger.warning("It ignores the requests that were created after the current embedding was created")

            query = """
                select
                    a.batchid, a.modelid, a.requestdtime as creationdtime,
                    b.domainid, b.customid, b.seq, b.classid, b.mastid, b.sentence
                from batch_req a
                inner join batch_req_detail b
                    on a.batchid = b.batchid
                        and a.modelid = b.modelid
                left outer join mast_embedding me
                    on b.domainid = me.domainid
                        and b.classid = me.classid
                        and b.mastid = me.mastid
                        and b.modelid = me.modelid
                left outer join class_embedding ce
                    on b.domainid = ce.domainid
                        and b.classid = ce.classid
                        and b.mastid is NULL
                        and b.modelid = ce.modelid
                where a.batchid = :batchid
                    and a.modelid = :modelid
                    and b.customid in :customids
                    and (me.creationdtime is null or me.creationdtime < a.requestdtime)
                    and (ce.creationdtime is null or ce.creationdtime < a.requestdtime)
            """

            params = {
                'batchid': batchid,
                'modelid': EMBEDDING_MODEL_ID,
                'customids': tuple(records['customid'].tolist())
            }

            req_details = db.execute(query, params)

            if req_details.empty:
                logging.warning(f"No metadata found for the batch. Skipping...")
                continue

            req_details['seq'] = req_details['seq'].astype(int)
            req_details['creationdtime'] = pd.to_datetime(req_details['creationdtime'])
            records = pd.merge(records, req_details, on=['domainid', 'modelid', 'batchid', 'customid', 'seq'], how='inner')
            records = records[['modelid', 'domainid', 'classid', 'mastid', 'sentence', 'embedding', 'creationdtime']]

            logger.info("Inserting embedding data into the database")
            update_embeddings(records, domainid, EMBEDDING_MODEL_ID, dimension)
        
        logger.info("Deleting batch request details")
        query = "delete from batch_req_detail where batchid = :batchid and modelid = :modelid"
        db.execute(query, params={'batchid': batchid, 'modelid': EMBEDDING_MODEL_ID})

        logger.info("Deleting input / output / error files")
        delete_file_on_batch_request(openai_client, batch_status.input_file_id)
        delete_file_on_batch_request(openai_client, batch_status.output_file_id)
        delete_file_on_batch_request(openai_client, batch_status.error_file_id)
        
        logger.info("Update batch status to COMPLETED")
        query = "update batch_req set status = :status where batchid = :batchid and modelid = :modelid"
        db.execute(query, params={'batchid': batchid, 'status': 'COMPLETED', 'modelid': EMBEDDING_MODEL_ID})
    end_time = time.time()
    logger.info(f'======================= process_batch_response end {end_time - start_time} sec =======================')

doc_md = """
    요청 중인 OpenAI Batch를 확인하고, 완료된 Batch의 결과를 처리합니다.
    Airflow의 Connection 접속정보가 사전에 정의되어 있어야 합니다.
    Prerequisite : Airflow Connection Info (tobeway_mdm, tobeway_aiml)

    Params {
        domainid : 대상 도메인 (전체: None)
    }
    
    1. listen_to_responses : OpenAI Batch API를 통해 요청된 Batch의 상태를 확인합니다.
    2. process_batch_response : 완료된 Batch의 결과를 처리합니다.
"""

with DAG(
    dag_id="aiml_embedding_batch_vca1_response",
    schedule=None,
    start_date=dt.datetime(2024, 4, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    catchup=False,
    params={
        "domainid": Param("ALL", type="string", enum=DOMAINIDS + ["ALL"]),
    },
    doc_md=doc_md,
    max_active_runs=1,
) as dag:
    batchids_list = listen_to_responses(domainid="{{ params.domainid }}")
    process_batch_response.expand(batchids=batchids_list)