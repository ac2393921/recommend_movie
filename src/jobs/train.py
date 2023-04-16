from loguru import logger

from src.models.base_recommender import BaseRecommender
from src.models.dataset import Dataset
from src.models.eval import MetricCaluculator, Metrics
from src.models.recommend_result import RecommendResult


class Train:
    def train(self, model: BaseRecommender, movies: Dataset) -> RecommendResult:
        logger.info("start train")
        recommend_result = model.recommend(movies)
        return recommend_result

    def evaluate(
        self,
        movies: Dataset,
        recommend_result: RecommendResult,
    ) -> Metrics:
        logger.info("start evaluation")
        metrics = MetricCaluculator().calc(
            movies.test.rating.tolist(),
            recommend_result.rating.tolist(),
            movies.test_user2items,
            recommend_result.user2items,
            k=10,
        )

        return metrics

    def train_and_evaluate(
        self,
        model: BaseRecommender,
        movies: Dataset,
    ) -> Metrics:
        logger.info("start training and evaluation")
        recommend_result = self.train(
            model=model,
            movies=movies,
        )

        evaluation = self.evaluate(
            movies=movies,
            recommend_result=recommend_result,
        )

        logger.info("done training and evaluation")
        return evaluation
