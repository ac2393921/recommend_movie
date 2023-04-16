from loguru import logger

from src.jobs.retrieve import DataLoader
from src.jobs.train import Train
from src.models.random_recommender import RandomRecommender

if __name__ == "__main__":
    logger.info("Start main.py")
    data_loader = DataLoader(
        num_users=1000, num_test_items=5, data_path="data/ml-10m/ml-10M100K"
    )
    movies = data_loader.load_data()
    logger.info("finish data loader")

    logger.info("start recommend")
    recommender = RandomRecommender()
    train = Train()
    metrics = train.train_and_evaluate(model=recommender, movies=movies)

    logger.info(f"metrics: {metrics}")
