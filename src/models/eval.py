from typing import Dict, List

import numpy as np
from pydantic import BaseModel


class Metrics(BaseModel):
    """評価指標"""

    rmse: float
    precision_at_k: float
    recall_at_k: float


class MetricCaluculator:
    def calc(
        self,
        true_rating: List[float],
        pred_rating: List[float],
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> Metrics:
        """指標を計算する

        Args:
            true_rating (List[float]): 真の評価値
            pred_rating (List[float]): 実際の評価値
            true_user2items (Dict[int, List[int]]): 真のユーザーとアイテムの対応
            pred_user2items (Dict[int, List[int]]): 実際のユーザーとアイテムの対応
            k (int): レコメンド数

        Returns:
            Metrics: 評価指標
        """
        rmse = self._calc_rmse(true_rating, pred_rating)
        precision_at_k = self._calc_precision_at_k(true_user2items, pred_user2items, k)
        recall_at_k = self._calc_recall_at_k(true_user2items, pred_user2items, k)

        return Metrics(rmse, precision_at_k, recall_at_k)

    def _precision_at_k(
        self, true_items: Dict[int, List[int]], pred_items: Dict[int, List[int]], k: int
    ) -> float:
        """Precision@kを計算する

        Args:
            true_items (Dict[int, List[int]]): 真にレコメンドされるアイテム
            pred_items (Dict[int, List[int]]): 実際にレコメンドされたアイテム
            k (int): レコメンド数

        Returns:
            float: Precision@k
        """
        if k == 0:
            return 0.0
        p_at_k = len(set(true_items) & set(pred_items[:k])) / k

        return p_at_k

    def _recall_at_k(
        self, true_items: List[int], pred_items: List[int], k: int
    ) -> float:
        """Recall@kを計算する

        Args:
            true_items (List[int]): 真にレコメンドされるアイテム
            pred_items (List[int]): 実際にレコメンドされたアイテム
            k (int): レコメンド数

        Returns:
            float: Recall@k
        """
        if len(true_items) == 0 or k == 0:
            return 0.0
        r_at_k = len(set(true_items) & set(pred_items[:k])) / len(true_items)

        return r_at_k

    def _calc_rmse(
        self,
        true_rating: List[float],
        pred_rating: List[float],
    ) -> float:
        """RMSEを計算する

        Args:
            true_rating (List[float]): 真の評価値
            pred_rating (List[float]): 実際の評価値

        Returns:
            float: RMSE
        """
        return np.sqrt(np.mean((np.array(true_rating) - np.array(pred_rating)) ** 2))

    def _calc_recall_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        """Recall@kを計算する

        Args:
            true_user2items (Dict[int, List[int]]): 真のユーザーとアイテムの対応
            pred_user2items (Dict[int, List[int]]): 実際のユーザーとアイテムの対応
            k (int): レコメンド数

        Returns:
            float: Recall@k
        """
        scores = []
        for user_id in true_user2items.keys():
            r_at_k = self._recall_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(r_at_k)

        return np.mean(scores)

    def _calc_precision_at_k(
        self,
        true_user2items: Dict[int, List[int]],
        pred_user2items: Dict[int, List[int]],
        k: int,
    ) -> float:
        """Precision@kを計算する

        Args:
            true_user2items (Dict[int, List[int]]): 真のユーザーとアイテムの対応
            pred_user2items (Dict[int, List[int]]): 実際のユーザーとアイテムの対応
            k (int): レコメンド数

        Returns:
            float: Precision@k
        """
        scores = []
        for user_id in true_user2items.keys():
            p_at_k = self._precision_at_k(
                true_user2items[user_id], pred_user2items[user_id], k
            )
            scores.append(p_at_k)

        return np.mean(scores)
