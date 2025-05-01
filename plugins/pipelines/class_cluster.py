import time
import nltk
import logging
import re
import faiss
import numpy as np

from tbwlib.utils import database as db
from tbwlib.mdm import get_dimension, is_leaf_class

from typing import List
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import normalize

from airflow.decorators import task
from airflow.exceptions import AirflowFailException, AirflowSkipException

logger = logging.getLogger(__name__)

DOMAINIDS = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
class FaissKMeansClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, dimension, max_iter=300, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self._labels = None
        self._dtype = np.float32 if dimension <= 2000 else np.float16

    def fit(self, X: List[List[float]]):
        X_normalized = np.array(normalize([np.array(x, dtype=self._dtype) for x in X]))

        self.clusterer = faiss.Kmeans(d=X_normalized.shape[1], k=self.n_clusters, niter=self.max_iter, verbose=self.verbose)
        self.clusterer.train(X_normalized)

        _, self._labels = self.clusterer.index.search(X_normalized, 1)
        self._labels = self._labels.flatten()
        return self

    def predict(self, X: List[List[float]]):
        assert self.clusterer is not None, "Model must be fitted before calling predict."
        X_normalized = np.array(normalize([np.array(x, dtype=self._dtype) for x in X]))

        _, labels = self.clusterer.index.search(X_normalized, 1)
        return labels.flatten()

    @property
    def labels(self):
        return self._labels.flatten() if self._labels is not None else None

def create_topics(sentences: List[str], n_topics: int) -> str:
    pattern = r'-[^:]+:(.*?)<br>'

    documents = [re.sub(pattern, r'\1', doc) for doc in sentences]
    stop_words = stopwords.words('english')
    sentences = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in documents]

    id2word = corpora.Dictionary(sentences)
    corpus = [id2word.doc2bow(text) for text in sentences]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        per_word_topics=True,
        eval_every=5
    )

    words = {}
    for i in range(len(lda_model.get_topics())):
        topic = lda_model.show_topic(i)
        for word, prob in topic:
            pattern = re.compile('^[a-zA-Z0-9]*$')
            matched = pattern.match(word)
            if (matched and len(word) > 2) or (not matched and len(word) > 1):
                words[word] = words.get(word, 0) + prob
    
    words = sorted(words, key=lambda k: words[k], reverse=True)

    num_word = min(n_topics, len(words))

    return ', '.join(words[:num_word])

@task
def generate_topics_for_classes(embedding_model_id: str, domainid: str, classid: str, n_topics: str):
    start_time = time.time()
    logger.info('======================= get_topics_for_classes start =======================')

    embedding_model_id = embedding_model_id.upper()
    domainid = domainid.upper()
    classid = classid.upper() if classid != 'None' and classid else None
    n_topics = int(n_topics)

    if domainid not in DOMAINIDS:
        end_time = time.time()
        logger.info(f"======================= get_topics_for_classes fail {end_time - start_time} sec =======================")
        raise ValueError(f"Invalid domainid: {domainid}. Allowed values are: {DOMAINIDS}")

    dimension = get_dimension(embedding_model_id)
    if not dimension:
        end_time = time.time()
        logger.info(f"======================= get_topics_for_classes fail {end_time - start_time} sec =======================")
        raise AirflowFailException(f"Dimension not found for embedding model ID: {embedding_model_id}")

    if classid and not is_leaf_class(domainid, classid):
        end_time = time.time()
        logger.info(f"======================= get_topics_for_classes fail {end_time - start_time} sec =======================")
        raise ValueError(f"Invalid classid: {classid}. It must be a leaf class if provided.")
    
    nltk.download('stopwords')

    class_cond = f"and m.classid = :classid" if classid else ""

    query = f"""
        with ranked as (
            select m.mastid, m.sentence, m.classid, (m.embedding_{dimension} <=> cm.avg_embedding_{dimension}) as distance,
                row_number() over (partition by m.classid order by (m.embedding_{dimension} <=> cm.avg_embedding_{dimension})) as rn
            from mast_embedding m
            inner join class_embedding cm
                on m.domainid = cm.domainid
                    and m.modelid = cm.modelid
                    and m.classid = cm.classid
            where m.domainid = :domainid
                and m.modelid = :modelid
                {class_cond}
        )
        select r.classid, r.sentence
        from ranked r
        where r.rn <= 30
    """

    params = {
        'domainid': domainid,
        'modelid': embedding_model_id,
        'classid': classid
    }
    records = db.execute(query, params)

    if records.empty:
        end_time = time.time()
        logger.info(f"======================= get_topics_for_classes end {end_time - start_time} sec =======================")
        raise AirflowSkipException(f"No records found for domainid: {domainid}, classid: {classid}")

    records = records.groupby('classid')['sentence'].agg(list).reset_index()
    records['topics'] = None

    for i, row in records.iterrows():
        classid = row['classid']
        sentences = row['sentence']
        try:
            topics = create_topics(sentences, n_topics)
            records.at[i, 'topics'] = topics.strip()
        except Exception as e:
            logger.error(f"Error creating topics for classid {classid}: {e}")
    
    records['topics'] = records['topics'].where(records['topics'].notnull() & (records['topics'] != ''), None)
    records = records[['classid', 'topics']]

    records[['modelid', 'domainid']] = embedding_model_id, domainid
    db.upsert_df(records, 'class_embedding', match_columns=['domainid', 'modelid', 'classid'])

    end_time = time.time()
    logger.info(f"======================= get_topics_for_classes end {end_time - start_time} sec =======================")