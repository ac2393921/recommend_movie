from collections import defaultdict

import numpy as np
from loguru import logger

from src.models.base_recommender import BaseRecommender
from src.models.dataset import Dataset
from src.models.recommend_result import RecommendResult


class PopularityRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # 評価値の閾値
        minimum_num_rating = kwargs.get("minimum_num_rating", 200)
        # 各アイテムごとの平均の評価値を計算し、その平均評価値を予測値とする。

        movie_rating_average = dataset.train.groupby("movie_id").agg(
            {"rating": np.mean}
        )

        # テストデータに予測値を格納する。
        # テストデータのみに存在するアイテムの予測値評価は0とする。
        movie_rating_predict = dataset.test.merge(
            movie_rating_average, on="movie_id", how="left", suffixes=("_test", "_pred")
        ).fillna(0)
        # 各ユーザーに対するおすすめ映画は、そのユーザーがまだ評価していない映画の中で、
        # 評価値が高いもの10作品とする。
        # ただし、評価値が閾値以上のもののみを対象とする。
        pred_user2items = defaultdict(list)
        user_watched_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        movie_stats = dataset.train.groupby("movie_id").agg(
            {"rating": [np.size, np.mean]}
        )
        atleast_flg = movie_stats["rating"]["size"] >= minimum_num_rating
        movies_sorted_by_rating = (
            movie_stats[atleast_flg]
            .sort_values(by=[("rating", "mean")], ascending=False)
            .index.tolist()
        )

        for user_id in dataset.train.user_id.unique():
            for movie_id in movies_sorted_by_rating:
                if movie_id not in user_watched_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                    if len(pred_user2items[user_id]) >= 10:
                        break

        return RecommendResult(
            rating=movie_rating_predict.rating_pred, user2items=pred_user2items
        )
