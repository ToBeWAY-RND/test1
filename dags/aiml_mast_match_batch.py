import time
import datetime as dt
import logging
import traceback
import json

from tbwlib.aiml import (
    MATCHING_OPTIONS,
    MATCH_CHECK_TYPES,
    Splink,
    validate_match_options,
    get_embedding_count
)

from tbwlib.mdm import (
    DOMAINIDS,
    delete_match,
    insert_match,
    get_masts_records,
    get_dimension,
    get_classes_with_splink_configs
)
from tbwlib.utils import database as db, pandas as pd

from splink.duckdb.linker import DuckDBLinker
from zoneinfo import ZoneInfo
from typing import List
from math import log2

from airflow import DAG
from airflow.models import Variable
from airflow.models.param import Param
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException, AirflowFailException

TIME_ZONE = Variable.get("TIME_ZONE")

TASK_SIZE = 3
BATCH_SIZE = 500

logger = logging.getLogger(__name__)

@task
def delete_current_matches(op_type: str, check_type: str, domainid: str, classid: str, matchingoption: str, max_match_count: str) -> str:
    start_time = time.time()
    logger.info('======================= delete_current_matches start =======================')
    op_type = op_type.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    check_type = check_type.upper()
    matchingoption = matchingoption.upper()
    max_match_count = int(max_match_count) if max_match_count != 'None' and max_match_count else 0
    
    error_msg = validate_match_options(op_type, domainid, check_type, matchingoption)
    if error_msg:
        end_time = time.time()
        logger.info(f'======================= delete_current_matches fail {end_time - start_time} sec =======================')
        raise AirflowFailException(error_msg)

    logger.info("Deleting current match results")
    delete_match(op_type, domainid, classid, check_type, max_match_count)

    end_time = time.time()
    logger.info(f'======================= delete_current_matches end {end_time - start_time} sec =======================')
    return check_type

@task
def get_classids_for_matching(op_type: str, embedding_model_id: str, domainid: str, classid: str, matchingoption: str, dup_score: str, min_sentence_length: str) -> List[List[str]]:
    start_time = time.time()
    logger.info('======================= get_classids_for_matching start =======================')

    op_type = op_type.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    matchingoption = matchingoption.upper()
    embedding_model_id = embedding_model_id.upper()
    dup_score = float(dup_score) if dup_score != 'None' and dup_score else None
    min_sentence_length = int(min_sentence_length) if min_sentence_length != 'None' and min_sentence_length else 0

    if embedding_model_id == 'SPLK':
        raise AirflowSkipException('Running Splink match, skipping Embedding match')

    if op_type == 'MATCH' and dup_score is None:
        raise AirflowFailException("Dup Score must be provided for MATCH operation.")
    if min_sentence_length < 0:
        raise AirflowFailException("Minimum sentence length must be a non-negative integer.")
    
    dimension = get_dimension(embedding_model_id)
    
    if dimension is None:
        end_time = time.time()
        logger.info(f'======================= process_match fail {end_time - start_time} sec =======================')
        raise AirflowFailException(f"Failed to get dimension for {embedding_model_id}.")

    query = "delete from embedding_queue where op_type = :op_type"
    db.execute(query, params={'op_type': op_type})

    matchclassid = classid if matchingoption == 'DOMAIN' else None
    exempt_ids = get_masts_records(domainid, matchclassid, exempt_only=True).drop(columns=['nomatch'])
    exempt_ids[['modelid', 'op_type', 'batchid']] = embedding_model_id, op_type, 'batch_match_exempt'

    logger.info("Saving exempt IDs to embedding queue")
    db.insert_df(exempt_ids, 'embedding_queue')

    records: pd.DataFrame = get_embedding_count(domainid, embedding_model_id, parent_classid=classid)

    if records.empty:
        end_time = time.time()
        logger.info(f"======================= get_classes_for_clustering end {end_time - start_time} sec =======================")
        raise AirflowSkipException(f"No records to match for {domainid} {embedding_model_id} under {classid}")

    records['group'] = pd.partition_with_lpt(records, TASK_SIZE, BATCH_SIZE, 'cnt') 
    classids_list = records.groupby('group')['classid'].agg(list).tolist()

    end_time = time.time()
    logger.info(f'======================= get_classids_for_matching end {end_time - start_time} sec =======================')
    return classids_list

@task
def process_match(op_type: str, embedding_model_id: str, domainid: str, classids: List[str], matchingoption: str, match_score: str, max_match_count: str, dup_score: str, min_sentence_length: str) -> bool:
    start_time = time.time()
    logger.info('======================= process_match start =======================')

    op_type = op_type.upper()
    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    matchingoption = matchingoption.upper()
    match_score = float(match_score)
    max_match_count = int(max_match_count)
    dup_score = float(dup_score) if dup_score != 'None' and dup_score else None
    min_sentence_length = int(min_sentence_length) if min_sentence_length != 'None' and min_sentence_length else 0
    try:
        dimension = get_dimension(embedding_model_id)
        total_match_count = 0

        for classid in classids:
            logger.info(f"Processing match for classid: {classid}")
            query = """
                select mastid
                from mast_embedding
                where modelid = :modelid and domainid = :domainid and classid = :classid
                    and length(sentence) >= :min_sentence_length
            """
            params = {
                'modelid': embedding_model_id,
                'domainid': domainid,
                'classid': classid,
                'min_sentence_length': min_sentence_length
            }
            mastids = db.execute(query, params=params)['mastid'].tolist()

            for i in range(0, len(mastids), BATCH_SIZE):
                logger.info(f"Processing match for mastids batch: {i // BATCH_SIZE + 1} / {len(mastids) // BATCH_SIZE + 1}")
                batch_mastids = mastids[i:i + BATCH_SIZE]

                if matchingoption == 'DOMAIN':
                    join_condition = f"and c.domainid = q.domainid"
                else:  # matchingoption == 'CLASS'
                    join_condition = f"and c.domainid = q.domainid and c.classid = q.classid"

                query = f"""
                    with record as (
                        select modelid, domainid, classid, mastid, embedding_{dimension} as embedding
                        from mast_embedding m
                        where modelid = :modelid
                            and domainid = :domainid
                            and classid = :classid
                            and mastid in :mastids
                    ),
                    exempt as (
                        select domainid, classid, mastid
                        from embedding_queue
                        where op_type = :op_type
                    )
                    select * from
                    (
                        select q.domainid, q.classid, q.mastid,
                            c.domainid as matchdomainid, c.classid as matchclassid, c.mastid as matchmastid,
                            trunc(100.0 - (c.distance::numeric * 100.0), 4) as matchscore
                        from record q
                        join lateral (
                            select c.domainid, c.classid, c.mastid, c.embedding_{dimension} <=> q.embedding as distance
                            from mast_embedding c
                            where c.modelid = q.modelid
                                and length(c.sentence) >= :min_sentence_length
                                {join_condition}
                                and c.mastid <> q.mastid
                                and (c.domainid, c.classid, c.mastid) not in (select domainid, classid, mastid from exempt)
                            order by q.embedding <=> c.embedding_{dimension}
                            limit :max_count
                        ) c on true
                        where (q.domainid, q.classid, q.mastid) not in (select domainid, classid, mastid from exempt)
                    ) x
                    where x.matchscore >= :match_score
                """

                params = {
                    'modelid': embedding_model_id,
                    'domainid': domainid,
                    'classid': classid,
                    'mastids': tuple(batch_mastids),
                    'op_type': op_type,
                    'max_count': max_match_count,
                    'match_score': match_score,
                    'min_sentence_length': min_sentence_length
                }
                records = db.execute(query, params=params)

                if records.empty:
                    logger.info(f"No matches found for this batch. Skipping...")
                    continue
                    
                logger.info(f"Found {len(records)} matches for this batch. Saving to database...")
                total_match_count += len(records)

                if dup_score is not None:
                    records['dupscore'] = dup_score
                
                records['creationdtime'] = pd.Timestamp.now(tz=TIME_ZONE).to_pydatetime()

                insert_match(op_type, embedding_model_id, domainid, records, max_match_count)

        logger.info(f"Total matches processed: {total_match_count}")
    except Exception as e:
        traceback.print_exc()
        return False
        
    end_time = time.time()
    logger.info(f'======================= process_match end {end_time - start_time} sec =======================')
    return True

@task
def cleanup_exempt_ids(op_type: str, _: List[bool]):
    start_time = time.time()
    logger.info('======================= cleanup_exempt_ids start =======================')

    op_type = op_type.upper()

    logger.info("Cleaning up exempt IDs from embedding queue")
    query = "delete from embedding_queue where op_type = :op_type"
    db.execute(query, params={'op_type': op_type})

    end_time = time.time()
    logger.info(f'======================= cleanup_exempt_ids end {end_time - start_time} sec =======================')

@task
def delete_stale_splink_model(check_type: str):
    start_time = time.time()
    logger.info('======================= delete_stale_splink_model start =======================')

    check_type = check_type.upper()

    if check_type != 'SPLK':
        raise AirflowSkipException("Not a Splink match operation. Skipping...")

    query = "select dictionaryid as domainid, classid from tobecontenti.mastmatchconfig where matchtype = 'SPLK'"
    configs = db.execute(query, db_name='MDM_DB')

    query = "select domainid, classid from splink_model"
    models = db.execute(query)

    models = pd.merge(models, configs, on=['domainid', 'classid'], how='leftanti')
    if models.empty:
        logger.info("No stale Splink models found. Skipping...")
        end_time = time.time()
        logger.info(f'======================= delete_stale_splink_model end {end_time - start_time} sec =======================')
        return

    db.delete_df(models, 'splink_model')

    end_time = time.time()
    logger.info(f'======================= delete_stale_splink_model end {end_time - start_time} sec =======================')

@task
def process_splink_match(op_type:str, check_type: str, domainid: str, classid: str, matchingoption: str, match_score: str, dup_score: str, max_match_count: str, lang: str, load_model: str) -> bool:

    start_time = time.time()
    logger.info('======================= process_splink_match start =======================')

    check_type = check_type.upper()
    op_type = op_type.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    matchingoption = matchingoption.upper()
    match_score = float(match_score)
    dup_score = float(dup_score) if dup_score != 'None' and dup_score else None
    max_match_count = int(max_match_count) if max_match_count != 'None' and max_match_count else 0
    load_model = load_model == 'True' or load_model == True

    if check_type != 'SPLK':
        raise AirflowSkipException("Not a Splink match operation. Skipping...")

    if op_type == 'MATCH' and dup_score is None:
        raise AirflowFailException("Dup Score must be provided for MATCH operation.")

    inheritances = get_classes_with_splink_configs(domainid, classid)

    if not inheritances:
        if classid:
            raise AirflowFailException(f"No Splink Config found for domain {domainid} under classid: {classid}")
        else:
            raise AirflowSkipException(f"No classes with Splink Configs found for domainid: {domainid}")

    total_match_count = 0

    for model_classid, classids in inheritances.items():
        logger.info(f"Processing Splink match with config of {model_classid} for {len(classids)} classes")
        
        splink_model = Splink(domainid, model_classid, lang, load_model=load_model)
        
        if not load_model:
            logger.info("Training Splink model with records")
            splink_model.fit()
        
        query = """
            select classid, mastid, prop_values
            from splink_table
            where domainid = :domainid and model_classid = :model_classid
        """

        if matchingoption == 'CLASS':
            query += " and classid in :classids"

        params = {
            'domainid': domainid,
            'model_classid': model_classid,
            'classids': tuple(classids) if matchingoption == 'CLASS' else None
        }
        records = db.execute(query, params=params)
        
        records['prop_values'] = records['prop_values'].apply(json.loads)
        props = pd.json_normalize(records['prop_values'])
        records = pd.concat([records.drop(columns=['prop_values']), props], axis=1, ignore_index=False)
        records['unique_id'] = records['classid'] + '_' + records['mastid']
        
        records = splink_model.predict(
            records=records,
            filter_records=pd.DataFrame({'classid': classids}, dtype='object'),
            matchingoption=matchingoption,
            match_score=match_score,
            max_match_count=max_match_count,
            with_waterfall=True
        )

        if records.empty:
            logger.info(f"No matches found for classid: {model_classid}. Skipping...")
            continue
            
        splink_model.close()

        total_match_count += len(records)
        logger.info(f"Found {len(records)} matches for classid: {model_classid}. Saving to database...")

        bf_columns = [col for col in records.columns if 'bf_' in col]    
        tf_bf_columns = [col for col in bf_columns if 'bf_tf_adj_' in col]

        for tf_col in tf_bf_columns:
            bf_col = tf_col.replace('_tf_adj', '')
            records[bf_col] = records[bf_col] * records[tf_col]

        def get_cumulative_prob(weight_sum):
            return 1 / (1 + 2 ** (-weight_sum))

        def truncate_float(value, digit=5):
            value *= 100.0
            return float(f"{value:.{digit}f}")

        def get_waterfall_dict(row):
            weight_sum = splink_model._base_weight
            prev_prob = get_cumulative_prob(weight_sum)
            waterfall_dict = {'base_weight_prob': truncate_float(prev_prob)}
            
            for bf_col in bf_columns:
                if bf_col in tf_bf_columns:
                    continue
                col_name = bf_col.replace('bf_', '')
                weight_sum += log2(row[bf_col])
                curr_prob = get_cumulative_prob(weight_sum)
                waterfall_dict[col_name] = truncate_float(curr_prob - prev_prob)
                prev_prob = curr_prob
            return waterfall_dict

        records['matchjson'] = records.apply(get_waterfall_dict, axis=1)

        records = records[['mastid', 'classid', 'matchclassid', 'matchmastid', 'matchscore', 'matchjson']]

        records['dupscore'] = dup_score
        records['creationdtime'] = pd.Timestamp.now(tz=TIME_ZONE).to_pydatetime()
        records['domainid'] = domainid
        records['matchdomainid'] = domainid

        insert_match(op_type, check_type, domainid, records, max_match_count)
    
    logger.info(f"Total matches processed: {total_match_count}")

    end_time = time.time()
    logger.info(f'======================= process_splink_match end {end_time - start_time} sec =======================')
    return True

doc_md = """
    도메인 별 관련품 생성을 하기 위한 Job입니다.
    Airflow의 Connection 접속정보가 사전에 정의되어 있어야 합니다.
    Prerequisite : Airflow Connection Info (tobeway_aiml, tobeway_mdm)
    Params {
        op_type: 작업 종류 (MATCH, REF)
        domainid : 대상 도메인,
        classid : 대상 시작 클래스 id,
        check_type : 검사 방법,
        matchingoption : 도메인, 클래스별 매칭 여부,
        match_score : 매칭 스코어,
        max_match_count : 매칭 결과 최대 개수,
        min_sentence_length : 최소 문장 길이,
        default_lang : 기본 언어,
        load_model : 모델 로드 여부
    }

    1. delete_current_matches : 현재 매칭 결과 삭제
    2-1. get_classids_for_matching : 매칭을 위한 클래스 id 목록 가져오기
    3. process_match : 매칭 수행
    4. cleanup_exempt_ids : 매칭 제외 id 삭제
    2-2. process_splink_match : Splink 매칭 수행 
"""

with DAG(
    dag_id="aiml_mast_match_batch",
    schedule=None,
    start_date=dt.datetime(2024, 4, 1, tzinfo=ZoneInfo(TIME_ZONE)),
    catchup=False,
    params={
        "op_type": Param('MATCH', type="string", enum=["MATCH", "REF"]),
        "domainid": Param('MATERIAL', type="string", enum=DOMAINIDS),
        "classid" : Param(None, type=["string", "null"]),
        "check_type": Param('VCA1', type="string", enum=MATCH_CHECK_TYPES),
        "matchingoption": Param("DOMAIN", type="string", enum=MATCHING_OPTIONS),
        "match_score": Param(99.99, type="number"),
        "dup_score": Param(100, type=["number", "null"]),
        "max_match_count": Param(3, type="integer"),
        "min_sentence_length": Param(0, type=["integer", "null"]),
        "default_lang": Param('KO', type="string"),
        "load_model": Param(False, type="boolean")
    },
    doc_md=doc_md,
    max_active_runs=1,
) as dag:
    check_type = delete_current_matches(
        op_type="{{ params.op_type }}",
        check_type="{{ params.check_type }}",
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}",
        matchingoption="{{ params.matchingoption }}",
        max_match_count="{{ params.max_match_count }}",
    )

    # Embeddings
    classids_list = get_classids_for_matching(
        op_type="{{ params.op_type }}",
        embedding_model_id=check_type,
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}",
        matchingoption="{{ params.matchingoption }}",
        dup_score="{{ params.dup_score }}",
        min_sentence_length="{{ params.min_sentence_length }}"
    )
    flags = process_match.partial(
        op_type="{{ params.op_type }}",
        embedding_model_id=check_type,
        domainid="{{ params.domainid }}",
        matchingoption="{{ params.matchingoption }}",
        match_score="{{ params.match_score }}",
        max_match_count="{{ params.max_match_count }}",
        dup_score="{{ params.dup_score }}",
        min_sentence_length="{{ params.min_sentence_length }}"
    ).expand(classids=classids_list)

    cleanup_exempt_ids(
        op_type="{{ params.op_type }}",
        _=flags
    )

    # Splink
    delete_stale_splink_model(
        check_type=check_type
    )
    process_splink_match(
        op_type="{{ params.op_type }}",
        check_type=check_type,
        domainid="{{ params.domainid }}",
        classid="{{ params.classid }}",
        matchingoption="{{ params.matchingoption }}",
        match_score="{{ params.match_score }}",
        dup_score="{{ params.dup_score }}",
        max_match_count="{{ params.max_match_count }}",
        lang="{{ params.default_lang }}",
        load_model="{{ params.load_model }}"
    )