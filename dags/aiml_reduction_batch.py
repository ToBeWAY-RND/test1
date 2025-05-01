import datetime as dt
import logging
import json
import time
import numpy as np

from tbwlib.utils import database as db, pandas as pd
from tbwlib.mdm import (
    DOMAINIDS,
    get_dimension
)
from tbwlib.aiml import (
    EMBEDDING_MODELS,
    get_embedding_count
)

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from zoneinfo import ZoneInfo
from typing import List

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowFailException, AirflowSkipException

TIME_ZONE = Variable.get("TIME_ZONE")

BATCH_SIZE = 20000
TASK_SIZE = 3

# NUM_NEIGHBORS must be smaller than BATCH_SIZE
NUM_NEIGHBORS = 4

logger = logging.getLogger(__name__)

@task
def get_classids_for_reduction(embedding_model_id: str, domainid: str, classid: str) -> List[List[str]]:
    start_time = time.time()
    logger.info('======================= get_records_to_process start =======================')

    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None

    if embedding_model_id not in EMBEDDING_MODELS:
        end_time = time.time()
        logger.info(f'======================= get_classids_for_reduction fail {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Invalid embedding model ID: {embedding_model_id}. Allowed models are: {EMBEDDING_MODELS}")

    if domainid not in DOMAINIDS:
        end_time = time.time()
        logger.info(f'======================= get_classids_for_reduction fail {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Invalid domainid: {domainid}. Allowed domainids are: {DOMAINIDS}")

    records: pd.DataFrame = get_embedding_count(domainid, embedding_model_id, parent_classid=classid)

    if records.empty:
        end_time = time.time()
        logger.info(f"======================= get_classids_for_reduction end {end_time - start_time} sec =======================")
        raise AirflowSkipException(f"No records found for domainid: {domainid}, classid: {classid}")

    records['batchid'] = pd.partition_with_lpt(records, TASK_SIZE, BATCH_SIZE, 'cnt')
    
    logger.info(f"Found {len(records)} classids to process")

    classids_list = records.groupby('batchid')['classid'].agg(list).tolist()

    end_time = time.time()
    logger.info(f"======================= get_classids_for_reduction end {end_time - start_time} sec =======================")
    return classids_list


@task
def process_reduction_batch(embedding_model_id: str, domainid: str, classids: List[str]):
    start_time = time.time()
    logger.info('======================= process_reduction_batch start =======================')

    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()

    dimension = get_dimension(embedding_model_id)

    if not dimension:
        end_time = time.time()
        logger.info(f"======================= process_reduction_batch fail {end_time - start_time} sec =======================")
        raise AirflowFailException(f"Dimension not found for embedding model ID: {embedding_model_id}")

    for classid in classids:
        count = get_embedding_count(domainid, embedding_model_id, classid=classid)
        logger.info(f"Processing classid: {classid}, count: {count}")

        query = f"""
            select mastid, embedding_{dimension} as embedding
            from mast_embedding
            where domainid = :domainid
                and modelid = :modelid
                and classid = :classid
            order by random()
            limit :batch_size
        """
        params = {
            'domainid': domainid,
            'modelid': embedding_model_id,
            'classid': classid,
            'batch_size': BATCH_SIZE
        }

        records = db.execute(query, params)
        records['embedding'] = records['embedding'].apply(json.loads)
        dtype = np.float32 if dimension <= 2000 else np.float16
        embeddings = np.array(records['embedding'].tolist(), dtype=dtype)
        random_state = np.random.RandomState()
        umap2d, umap3d = None, None

        if count == 1:
            logger.info("Only one record found, skipping dimensionality reduction")
            records['dim_reduction'] = [[0, 0, 0]]
            records['dim_reduction2'] = [[0, 0]]
        elif count <= NUM_NEIGHBORS:
            logger.info("Using t-SNE for dimensionality reduction due to small number of records")           
            perplexity = count // 2
            tsne2d = TSNE(n_components=2, perplexity=perplexity, metric="precomputed", init="random", random_state=random_state, n_jobs=1)
            tsne3d = TSNE(n_components=3, perplexity=perplexity, metric="precomputed", init="random", random_state=random_state, n_jobs=1)

            distance_matrix = cosine_distances(embeddings)

            records['dim_reduction'] = tsne3d.fit_transform(distance_matrix).tolist()
            records['dim_reduction2'] = tsne2d.fit_transform(distance_matrix).tolist()

            records['dim_reduction'] = records['dim_reduction'].apply(list)
            records['dim_reduction2'] = records['dim_reduction2'].apply(list)
        else:
            logger.info("Using UMAP for dimensionality reduction")
            umap2d = UMAP(n_components=2, random_state=random_state, n_neighbors=NUM_NEIGHBORS, metric='cosine', n_jobs=1, min_dist=0.2)
            umap3d = UMAP(n_components=3, random_state=random_state, n_neighbors=NUM_NEIGHBORS, metric='cosine', n_jobs=1, min_dist=0.2)
            
            if count <= BATCH_SIZE:
                records['dim_reduction'] = umap3d.fit_transform(embeddings).tolist()
                records['dim_reduction2'] = umap2d.fit_transform(embeddings).tolist()

                records['dim_reduction'] = records['dim_reduction'].apply(list)
                records['dim_reduction2'] = records['dim_reduction2'].apply(list)
            else:
                umap2d.fit(embeddings)
                umap3d.fit(embeddings)
        
        records = records.drop(columns=['embedding'])

        if 'dim_reduction' in records.columns:
            logger.info(f"Saving reduction results for classid: {classid}")
            records[['modelid', 'domainid', 'classid']] = embedding_model_id, domainid, classid
            db.upsert_df(records, 'mast_embedding', match_columns=['domainid', 'modelid', 'classid', 'mastid'])
        
        else:  # count > BATCH_SIZE
            logger.info("Too many records, processing in chunks")
            query_fn = lambda mastid: f"""
                select mastid, embedding_{dimension} as embedding
                from mast_embedding
                where domainid = :domainid
                    and modelid = :modelid
                    and classid = :classid
                    {'and mastid > :mastid' if mastid else ''}
                order by mastid
                limit :batch_size
            """
            mastid = None

            for start_idx in range(0, count, BATCH_SIZE):
                params = {
                    'domainid': domainid,
                    'modelid': embedding_model_id,
                    'classid': classid,
                    'batch_size': BATCH_SIZE,
                    'mastid': mastid
                }

                records = db.execute(query_fn(mastid), params)
                if records.empty:
                    break

                records['embedding'] = records['embedding'].apply(json.loads)
                embeddings = np.array(records['embedding'].tolist(), dtype=dtype)

                records['dim_reduction'] = umap3d.transform(embeddings).tolist()
                records['dim_reduction2'] = umap2d.transform(embeddings).tolist()

                records['dim_reduction'] = records['dim_reduction'].apply(list)
                records['dim_reduction2'] = records['dim_reduction2'].apply(list)
                records = records.drop(columns=['embedding'])

                logger.info(f"Saving reduction results for classid: {classid} from {start_idx} to {min(start_idx + BATCH_SIZE, count)}")
                records = records[['mastid', 'dim_reduction', 'dim_reduction2']]
                records[['modelid', 'domainid', 'classid']] = embedding_model_id, domainid, classid
                db.upsert_df(records, 'mast_embedding', match_columns=['domainid', 'modelid', 'classid', 'mastid'])
                
                mastid = records['mastid'].iloc[-1]
        
    end_time = time.time()
    logger.info(f"======================= process_reduction_batch end {end_time - start_time} sec =======================")

doc_md = f"""
    클래스 별 임베딩을 UMAP으로 차원 축소하여 저장하는 DAG입니다.
    임베딩의 갯수가 NUM_NEIGHBORS(={NUM_NEIGHBORS})보다 작으면 t-SNE로 차원 축소합니다.
    1 개의 임베딩만 있는 경우는 차원 축소를 하지 않습니다.
    차원 축소된 결과는 `dim_reduction`과 `dim_reduction2` 컬럼에 저장됩니다.
    'aiml_embedding_batch' 나 'aiml_embedding_batch_response'와 동시에 실행하면 오류가 발생할 수 있습니다.

    Params {{
        embedding_model_id: 임베딩 모델 ID
        domainid: 대상 도메인
        classid: 대상 시작 클래스 ID (None이면 모든 클래스)
    }}
"""
with DAG(
    dag_id='aiml_reduction_batch',
    start_date = dt.datetime(2024, 1, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    schedule = None,
    doc_md=doc_md,
    catchup=False,
    params={
        "embedding_model_id": Param('VCA1', type="string", enum=EMBEDDING_MODELS),
        "domainid": Param('MATERIAL', type="string", enum=DOMAINIDS),
        "classid": Param(None, type=["string", "null"]),
    },
    max_active_runs=1,
) as dag:
    classids_list = get_classids_for_reduction(
        embedding_model_id="{{ params.embedding_model_id }}",
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}"
    )
    process_reduction_batch.partial(
        embedding_model_id="{{ params.embedding_model_id }}",
        domainid="{{ params.domainid }}"
    ).expand(classids=classids_list)
