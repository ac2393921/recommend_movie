from abc import ABC, abstractmethod

from loguru import logger

from src.models.dataset import Dataset
from src.models.eval import MetricCaluculator
from src.models.recommend_result import RecommendResult


class BaseRacommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self, movies: Dataset) -> None:
        recommend_result = self.recommend(movies)
        metrics = MetricCaluculator().calc(
            movies.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movies.test_user2items,
            recommend_result.user2items,
            k=10,
        )
        logger.info(f"metrics: {metrics}")
