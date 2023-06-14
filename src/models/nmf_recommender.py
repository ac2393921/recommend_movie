from collections import defaultdict

import faiss
import numpy as np
from loguru import logger
from sklearn.decomposition import NMF

from models.dataset import Dataset
from models.recommend_result import RecommendResult
from src.models.base_recommender import BaseRecommender
from src.models.dataset import Dataset
from src.models.recommend_result import RecommendResult


class NMFRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        fillna_with_zero = kwargs.get("fillna_with_zero", True)
        factors = kwargs.get("factors", 5)

        user_movie_matrix = dataset.train.pivot_table(
            index="user_id", columns="movie_id", values="rating"
        )
        user_id2index = dict(
            zip(user_movie_matrix.index, range(len(user_movie_matrix)))
        )
        movie_id2index = dict(
            zip(user_movie_matrix.columns, range(len(user_movie_matrix.columns)))
        )

        if fillna_with_zero:
            matrix = user_movie_matrix.fillna(0).to_numpy()
        else:
            matrix = user_movie_matrix.fillna(dataset.train.rating.mean()).to_numpy()

        nmf = NMF(n_components=factors)
        nmf.fit(matrix)
        P = nmf.fit_transform(matrix)
        Q = nmf.components_
        movie_mat = nmf.components_.T

        pred_matrix = np.dot(P, Q)
        average_score = dataset.train.rating.mean()

        movie_index = faiss.IndexFlatIP(5)
        logger.info(movie_mat.astype("float32"))
        movie_index.add(movie_mat.astype("float32"))
        faiss.write_index(movie_index, "features.index")
