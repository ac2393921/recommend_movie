from abc import ABC, abstractmethod

from src.models.dataset import Dataset
from src.models.recommend_result import RecommendResult


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass
