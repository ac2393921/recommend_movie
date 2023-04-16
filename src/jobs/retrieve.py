import os
from typing import Tuple

import pandas as pd
import pandera as pa
from loguru import logger
from pandera.typing import DataFrame

from src.dataset.shema import (
    MoviesBaseSchema,
    MoviesRatingSchema,
    MoviesSchema,
    RatingsBaseSchema,
    TagsBaseSchema,
)
from src.models.dataset import Dataset


class DataLoader:
    def __init__(
        self,
        num_users: int = 1000,
        num_test_items: int = 5,
        data_path: str = "../data/ml-10m/ml-10M100K/",
    ):
        self.num_users = num_users
        self.num_test_items = num_test_items
        self.data_path = data_path

    def load_data(self) -> Dataset:
        """データを読み込み、Datasetに変換する

        Returns:
            Dataset: データセット
        """
        logger.info("Start load data")
        ratings, movie_content = self._load()
        movie_train, movie_test = self._split_data(ratings)

        movie_test_user2items = (
            movie_test[movie_test.rating >= 4]
            .groupby("user_id")
            .agg({"movie_id": list})["movie_id"]
            .to_dict()
        )

        return Dataset(
            train=movie_train,
            test=movie_test,
            test_user2items=movie_test_user2items,
            item_content=movie_content,
        )

    @pa.check_types
    def _split_data(
        self, movies: DataFrame[MoviesSchema]
    ) -> Tuple[DataFrame[MoviesSchema], DataFrame[MoviesSchema]]:
        """データを学習用とテスト用に分割する

        Args:
            movies (DataFrame[MoviesSchema]): 映画データ

        Returns:
            Tuple[DataFrame[MoviesSchema], DataFrame[MoviesSchema]]: 学習用データ、テスト用データ
        """
        logger.info("Start split data")

        # 学習用とテスト用にデータを分割する
        # 各ユーザーの直近の映画5件を評価用に使い、それ以外を学習用とする
        # まずは、それぞれのユーザーが評した映画の順序を計算する
        # 直近付与した映画から順番を付与していく（0始まり）
        movies["rating_order"] = movies.groupby("user_id")["timestamp"].rank(
            method="first", ascending=False
        )

        movie_train = movies[movies.rating_order > self.num_test_items]
        movie_train = MoviesSchema(movie_train)
        movie_test = movies[movies.rating_order <= self.num_test_items]
        movie_train = MoviesSchema(movie_test)

        return movie_train, movie_test

    @pa.check_types
    def _load(self) -> Tuple[DataFrame[MoviesRatingSchema], DataFrame[MoviesSchema]]:
        """データを読み込む

        Returns:
            Tuple[DataFrame[MoviesRatingSchema], DataFrame[MoviesSchema]]: 映画評価データ、映画データ
        """
        movies = self._load_movies()
        ratings = self._load_ratings()

        # データを結合する
        logger.info("merge ratings data")
        movies_ratings = ratings.merge(movies, on="movie_id")
        movies_ratings = MoviesRatingSchema(movies_ratings)

        logger.info(movies_ratings.info())
        logger.info(movies_ratings.head())
        logger.info(movies.info())
        logger.info(movies.head())

        return movies_ratings, movies

    @pa.check_types
    def _load_movies(self) -> DataFrame[MoviesSchema]:
        """映画データを読み込む

        Returns:
            DataFrame[MoviesSchema]: 映画データ
        """
        # 映画の情報の読み込み
        logger.info("load movies data")
        m_cols = ["movie_id", "title", "genres"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            names=m_cols,
            sep="::",
            encoding="latin-1",
            engine="python",
        )

        # genreをlistを形式で保持する
        movies["genres"] = movies.genres.apply(lambda x: x.split("|"))
        movies = MoviesBaseSchema(movies)

        # ユーザーがタグ付けした映画の情報の読み込み
        logger.info("load tags data")
        t_cols = ["user_id", "movie_id", "tag", "timestamp"]
        user_tagged_movies = pd.read_csv(
            os.path.join(self.data_path, "tags.dat"),
            names=t_cols,
            sep="::",
            engine="python",
        )
        # tagを小文字にする
        user_tagged_movies["tag"] = user_tagged_movies["tag"].str.lower()
        user_tagged_movies = TagsBaseSchema(user_tagged_movies)
        movie_tags = user_tagged_movies.groupby("movie_id").agg({"tag": list})

        # タグ情報を映画情報に結合する
        movies = movies.merge(movie_tags, on="movie_id", how="left")
        movies = MoviesSchema(movies)

        return movies

    @pa.check_types
    def _load_ratings(self) -> DataFrame[RatingsBaseSchema]:
        """映画評価データを読み込む

        Returns:
            DataFrame[RatingsBaseSchema]: 映画評価データ
        """
        # ユーザーの評価情報の読み込み
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            names=r_cols,
            sep="::",
            encoding="latin-1",
            engine="python",
        )
        # ユーザー数をnum_usersに制限する
        valid_user_ids = sorted(ratings.user_id.unique())[: self.num_users]
        ratings = ratings[ratings.user_id.isin(valid_user_ids)]
        ratings = RatingsBaseSchema(ratings)

        return ratings
