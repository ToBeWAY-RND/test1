import time
import logging

from airflow.decorators import task
from airflow.exceptions import AirflowFailException

from typing import List
from tbwlib.mdm import (
    update_embedding_status,
    get_embedding_records
)
from tbwlib.utils import database as db

logger = logging.getLogger(__name__)


@task
def cleanup_stale_embeddings(embedding_model_id: str, domainid: str, allowed_models: List[str], allowed_domainids: List[str]):
    start_time = time.time()
    logger.info('======================= delete_stale_embeddings start =======================')
    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()

    if embedding_model_id not in allowed_models:
        end_time = time.time()
        logger.info(f'======================= delete_stale_embeddings fail {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Invalid embedding model ID: {embedding_model_id}. Allowed models are: {allowed_models}")

    if domainid not in allowed_domainids:
        end_time = time.time()
        logger.info(f'======================= delete_stale_embeddings fail {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Invalid domainid: {domainid}. Allowed domainids are: {allowed_domainids}")
    
    logger.info("Deleting orphan embeddings from embedding tables")
    records = get_embedding_records(embedding_model_id, domainid, status='D', include_class=True)

    if records.empty:
        logger.info("No orphan embeddings found")
    else:
        records = records.drop(columns=['status', 'sentence'])
        masts = records[records['mastid'].notnull()]
        classes = records[records['mastid'].isna()].drop(columns=['mastid'])
        if not masts.empty:
            db.delete_df(masts, 'mast_embedding')
        if not classes.empty:
            db.delete_df(classes, 'class_embedding')

    update_embedding_status(embedding_model_id, domainid, records, status='X')

    end_time = time.time()
    logger.info(f'======================= delete_stale_embeddings end {end_time - start_time} sec =======================')


