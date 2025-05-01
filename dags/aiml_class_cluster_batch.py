import logging
import time
import nltk

import numpy as np
import datetime as dt
import json

from pipelines.class_cluster import (
    generate_topics_for_classes
)

from tbwlib.utils import database as db, pandas as pd
from tbwlib.mdm import (
    DOMAINIDS,
    CODEI_DOMAINID,
    get_dimension,
    is_leaf_class,
    delete_cluster,
    insert_cluster
)
from tbwlib.aiml import (
    EMBEDDING_MODELS,
    get_embedding_count
)

from typing import List
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from zoneinfo import ZoneInfo
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowFailException, AirflowSkipException

logging.getLogger("gensim").propagate = False
logger = logging.getLogger(__name__)

N_CLUSTERS_MIN = 2
N_CLUSTERS_MAX = 10
N_TOPICS_MIN = 2
N_TOPICS_MAX = 10

TASK_SIZE = 3
BATCH_SIZE = 20000
MINIMUM_EMBEDDING_COUNT = 100
MINIMUM_CLUSTER_COUNT = 20

TIME_ZONE = Variable.get("TIME_ZONE")

@task
def get_classes_for_clustering(domainid: str, embedding_model_id: str, classid: str, n_clusters: str) -> List[List[str]]:
    start_time = time.time()
    logger.info('======================= get_classes_for_clustering start =======================')

    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    n_clusters = int(n_clusters) if n_clusters != 'None' and n_clusters else None

    if domainid not in DOMAINIDS:
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering fail {end_time - start_time} sec =======================")
        raise AirflowFailException(f"Invalid domainid: {domainid}. Allowed values are: {DOMAINIDS}")
    
    if domainid == CODEI_DOMAINID:
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering end {end_time - start_time} sec =======================")
        raise AirflowSkipException("Clustering is not supported for CODEI domain. Skipping...")

    if classid and not is_leaf_class(domainid, classid):
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering fail {end_time - start_time} sec =======================")
        raise AirflowFailException(f"Invalid classid: {classid}. It must be a leaf class if provided.")
    
    dimension = get_dimension(embedding_model_id)
    if not dimension:
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering fail {end_time - start_time} sec =======================")
        raise AirflowFailException(f"Dimension not found for embedding model ID: {embedding_model_id}")
    
    if not n_clusters or n_clusters < N_CLUSTERS_MIN or n_clusters > N_CLUSTERS_MAX:
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering fail {end_time - start_time} sec =======================")
        raise ValueError(f"Invalid n_clusters: {n_clusters}. Allowed range is [{N_CLUSTERS_MIN}, {N_CLUSTERS_MAX}]")
    
    class_cond = f"and classid = :classid" if classid else ""

    logger.info("Deleting existing clusters from class_cluster table")
    query = f"""
        delete from class_cluster
        where domainid = :domainid
            and modelid = :modelid
            {class_cond}
    """
    params = {
        'domainid': domainid,
        'modelid': embedding_model_id,
        'classid': classid
    }
    db.execute(query, params)

    logger.info("Resetting clusterid in mast_embedding table")
    query = f"""
        update mast_embedding
        set clusterid = null
        where domainid = :domainid
            and modelid = :modelid
            {class_cond}
    """
    db.execute(query, params)

    logger.info("Deleting existing clusters from MDM cluster table")
    delete_cluster(domainid, classid, embedding_model_id)

    records: pd.DataFrame = get_embedding_count(domainid, embedding_model_id, parent_classid=classid)

    if not records[records['cnt'] < MINIMUM_EMBEDDING_COUNT].empty:
        logger.warning("Classes with less than 100 embeddings will be skipped")

    records = records[records['cnt'] >= MINIMUM_EMBEDDING_COUNT]
    if records.empty:
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering end {end_time - start_time} sec =======================")
        raise AirflowSkipException(f"No records more than {MINIMUM_EMBEDDING_COUNT} found for domainid: {domainid}, classid: {classid}")

    records['group'] = pd.partition_with_lpt(records, TASK_SIZE, BATCH_SIZE, 'cnt') 
    classids_list = records.groupby('group')['classid'].agg(list).tolist()

    end_time = time.time()
    logger.info(f"======================= get_classes_for_clustering end {end_time - start_time} sec =======================")
    return classids_list

@task
def process_clustering_and_generate_topics(embedding_model_id: str, domainid: str, n_clusters: str, n_topics: str, classids: List[str]) -> None:
    start_time = time.time()
    logger.info('======================= process_clustering_and_generate_topics start =======================')

    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    n_clusters = int(n_clusters)
    n_topics = int(n_topics)
    dimension = get_dimension(embedding_model_id)

    nltk.download('stopwords')

    query = f"""
        select classid, count(*) as cnt
        from mast_embedding
        where domainid = :domainid
            and modelid = :modelid
        group by classid
    """
    params = {
        'domainid': domainid,
        'modelid': embedding_model_id
    }

    class_records = db.execute(query, params)
    class_records['cnt'] = class_records['cnt'].astype(int)
    class_records = class_records[class_records['classid'].isin(classids)]
    class_records['cohesiveness'] = None

    for i, row in class_records.iterrows():
        classid = row['classid']
        count = row['cnt']

        logger.info(f"Processing classid: {classid}")

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

        embeddings = records['embedding'].tolist()
        faiss_model = FaissKMeansClustering(n_clusters, dimension).fit(embeddings)
        class_records.loc[i, 'cohesiveness'] = round(silhouette_score(embeddings, faiss_model.labels), 2)

        dfs = []

        if count <= BATCH_SIZE:
            records['cluster'] = [str(x).zfill(2) for x in faiss_model.labels]
            dfs.append(records[['mastid', 'cluster']])
        else:
            logger.info(f"Clustering for classid {classid} with count {count} is too large. Processing in batches.")
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
            for _ in range(0, count, BATCH_SIZE):
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
                embeddings = records['embedding'].tolist()
                labels = faiss_model.predict(embeddings)
                records['cluster'] = [str(x).zfill(2) for x in labels]
                dfs.append(records[['mastid', 'cluster']])

                mastid = records['mastid'].iloc[-1]
        
        records = pd.concat(dfs)
        records['count'] = records.groupby('cluster')['mastid'].transform('count')
        records.loc[records['count'] < MINIMUM_CLUSTER_COUNT, 'cluster'] = 'UNCLUSTERED'

        unique_clusters = records.loc[records['cluster'] != 'UNCLUSTERED', 'cluster'].unique()
        remapping = {val: f"{i:02}" for i, val in enumerate(unique_clusters)}
        records['cluster'] = records['cluster'].replace(remapping)

        records = records.drop(columns=['count'])

        logger.info("Inserting clusters into MDM cluster table")
        insert_cluster(embedding_model_id, domainid, records)

        records['clusterid'] = records['cluster']
        records.loc[records['cluster'] != 'UNCLUSTERED', 'clusterid'] = classid + '_' + records['cluster']
        records = records.drop(columns=['cluster'])
        records[['modelid', 'domainid', 'classid']] = embedding_model_id, domainid, classid
        db.upsert_df(records, 'mast_embedding', match_columns=['domainid', 'modelid', 'classid', 'mastid'])

        logger.info("Calculating Centroids for clusters and inserting into class_cluster table")
        query = f"""
            select domainid, modelid, classid, clusterid, avg(embedding_{dimension}) as centroid
            from mast_embedding
            where domainid = :domainid
                and modelid = :modelid
                and classid = :classid
            group by domainid, modelid, classid, clusterid
        """
        params = {
            'domainid': domainid,
            'modelid': embedding_model_id,
            'classid': classid
        }
        records = db.execute(query, params)
        records[f'centroid'] = records[f'centroid'].apply(json.loads)
        records = records.rename(columns={'centroid': f'centroid_{dimension}'})

        db.insert_df(records, 'class_cluster')

        logger.info("Generating topics for clusters")
        query = f"""
            with ranked as (
                select m.mastid, m.sentence, m.classid, m.clusterid, (m.embedding_{dimension} <=> cc.centroid_{dimension}) as distance,
                    row_number() over (partition by m.clusterid order by (m.embedding_{dimension} <=> cc.centroid_{dimension})) as rn
                from mast_embedding m
                inner join class_cluster cc
                    on m.domainid = cc.domainid
                        and m.modelid = cc.modelid
                        and m.classid = cc.classid
                        and m.clusterid = cc.clusterid
                where m.domainid = :domainid
                    and m.modelid = :modelid
                    and m.classid = :classid
            )
            select r.clusterid, r.sentence
            from ranked r
            where r.rn <= 30
        """
        params = {
            'domainid': domainid,
            'modelid': embedding_model_id,
            'classid': classid
        }
        records = db.execute(query, params)
        records = records.groupby('clusterid')['sentence'].agg(list).reset_index()
        records['topics'] = None

        for i, row in records.iterrows():
            clusterid = row['clusterid']
            sentences = row['sentence']
            try:
                topics = create_topics(sentences, n_topics)
                records.at[i, 'topics'] = topics.strip()
            except Exception as e:
                logger.error(f"Error creating topics for clusterid {clusterid}: {e}")
        records = records[['clusterid', 'topics']]
        records[['modelid', 'domainid', 'classid']] = embedding_model_id, domainid, classid
        db.upsert_df(records, 'class_cluster', match_columns=['domainid', 'modelid', 'classid', 'clusterid'])
    
    logger.info("Saving cohesiveness to class_embedding table")
    class_records = class_records[['classid', 'cohesiveness']]
    class_records[['modelid', 'domainid']] = embedding_model_id, domainid
    db.upsert_df(class_records, 'class_embedding', match_columns=['domainid', 'modelid', 'classid'])

    end_time = time.time()
    logger.info(f"======================= process_clustering_and_generate_topics end {end_time - start_time} sec =======================")

doc_md = """
    분류 별 클러스터링을 하기 위한 Job입니다.
    Airflow의 Connection 접속정보가 사전에 정의되어 있어야 합니다.
    Prerequisite : Airflow Connection Info (tobeway_aiml)
    Params {
        domainid : 대상 도메인,
        embedding_model_id : 임베딩 모델 id,
        n_clusters : 클러스터 수,
        n_topics : 토픽 수,
        classid : 단일 대상 클래스 (None인 경우 전체 클래스 대상)
    }
    
"""

with DAG(
    'aiml_class_cluster_batch', 
    start_date = dt.datetime(2024, 2, 2, tzinfo=ZoneInfo(TIME_ZONE)),
    schedule = None,
    params={
        "domainid": Param('MATERIAL', type="string", enum=DOMAINIDS),
        "embedding_model_id": Param('VCA1', type="string", enum=EMBEDDING_MODELS),
        "n_clusters": Param(5, type=["integer", "null"]),
        "n_topics": Param(5, type="integer", minimum=N_TOPICS_MIN, maximum=N_TOPICS_MAX),
        "classid" : Param(None, type=["null", "string"])
    },
    catchup=False,
    max_active_runs=1,
    doc_md=doc_md
) as dag:
    generate_topics_for_classes(
        embedding_model_id="{{ params.embedding_model_id }}",
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}",
        n_topics="{{ params.n_topics }}"
    )

    classids_list = get_classes_for_clustering(
        domainid="{{ params.domainid }}",
        embedding_model_id="{{ params.embedding_model_id }}",
        classid="{{ params.classid }}",
        n_clusters="{{ params.n_clusters }}"
    )
    process_clustering_and_generate_topics.partial(
        embedding_model_id="{{ params.embedding_model_id }}",
        domainid="{{ params.domainid }}",
        n_clusters="{{ params.n_clusters }}",
        n_topics="{{ params.n_topics }}",
    ).expand(classids=classids_list)
