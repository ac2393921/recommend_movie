from collections import defaultdict

import numpy as np

from src.models.base_recommender import BaseRacommender
from src.models.dataset import Dataset
from src.models.recommend_result import RecommendResult


class RandomRecommender(BaseRacommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        """ランダムにレコメンドする

        Args:
            dataset (Dataset): データセット

        Returns:
            RecommendResult: レコメンド結果
        """

        # ユーザーIDとアイテムIDに対して、O始まりのインデックスを割り当てる
        unique_user_ids = sorted(dataset.train.user_id.unique())
        unique_movie_ids = sorted(dataset.train.movie_id.unique())
        # ユーザーIDとアイテムIDをインデックスに変換する
        user_id2index = {user_id: i for i, user_id in enumerate(unique_user_ids)}
        movie_id2index = {movie_id: i for i, movie_id in enumerate(unique_movie_ids)}

        # ユーザー×アイテムの行列で、各セルの予測評価値はO.5〜5.0の一様分布からサンプリングする
        pred_matrix = np.random.uniform(
            0.5, 5.0, (len(unique_user_ids), len(unique_movie_ids))
        )

        # RMSE評価用にテストデータに出てくるユーザーとアイテムの予測評価値を格納する
        movie_rating_predict = dataset.test.copy()
        pred_results = []
        for _, row in dataset.test.iterrows():
            user_id = row["user_id"]
            # テストデータのアイテムが学習データにない場合も乱数で予測する
            if row["user_id"] not in user_id2index:
                pred_results.append(np.random.uniform(0.5, 5.0))
                continue

            user_index = user_id2index[row["user_id"]]
            movie_index = movie_id2index[row["movie_id"]]
            pred_score = pred_matrix[user_index, movie_index]
            pred_results.append(pred_score)
        movie_rating_predict["rating"] = pred_results

        # ランキング評価用のデータを作成
        # 各ユーザーに対するおすすめ映画は、
        # そのユーザーがまだ評価していない映画の中からランダムに10作品を選ぶ
        # キーはユーザーIDで、値はおすすめの映画IDのリスト
        pred_user2items = defaultdict(list)
        # キーはユーザーIDで、値はおすすめの映画IDのリスト
        user_evaluated_movies = (
            dataset.train.groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )
        for user_id in unique_user_ids:
            user_index = user_id2index[user_id]
            movie_indexs = np.argsort(-pred_matrix[user_index, :])
            for movie_index in movie_indexs:
                movie_id = unique_movie_ids[movie_index]
                if movie_id not in user_evaluated_movies[user_id]:
                    pred_user2items[user_id].append(movie_id)
                    if len(pred_user2items[user_id]) == 10:
                        break

        return RecommendResult(
            rating=movie_rating_predict.rating, user2items=pred_user2items
        )


if __name__ == "__main__":
    RandomRecommender().run()
