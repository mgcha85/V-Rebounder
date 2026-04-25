from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class ModelConfig:
    model_type: str = "xgboost"
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 1.0
    random_state: int = 42


@dataclass
class TrainResult:
    model: Any
    feature_importance: dict[str, float] = field(default_factory=dict)
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)


class VReboundClassifier:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = None
        self.feature_cols: list[str] = []

    def train(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        label_col: str = "label",
        test_days: int = 365,
    ) -> TrainResult:
        self.feature_cols = feature_cols

        df_clean = df.drop_nulls(subset=feature_cols + [label_col])

        # Time-based split: test = latest test_days
        if "open_time" in df_clean.columns:
            max_time = df_clean["open_time"].max()
            if isinstance(max_time, datetime):
                split_time = max_time - timedelta(days=test_days)
            else:
                split_time = max_time - test_days * 24 * 60 * 60 * 1000
            train_df = df_clean.filter(pl.col("open_time") < split_time)
            test_df = df_clean.filter(pl.col("open_time") >= split_time)
        else:
            split_idx = int(len(df_clean) * (1 - test_days / 365 / 2))
            train_df = df_clean[:split_idx]
            test_df = df_clean[split_idx:]

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df.select(label_col).to_numpy().ravel()
        X_test = test_df.select(feature_cols).to_numpy()
        y_test = test_df.select(label_col).to_numpy().ravel()

        # Handle class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        if self.config.model_type == "xgboost" and HAS_XGB:
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif self.config.model_type == "lightgbm" and HAS_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config.random_state,
                verbose=-1,
            )
        else:
            raise ValueError(f"Model {self.config.model_type} not available")

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_proba = self.model.predict_proba(X_test)[:, 1]

        train_metrics = self._calc_metrics(y_train, train_pred)
        test_metrics = self._calc_metrics(y_test, test_pred)
        test_metrics["auc"] = self._calc_auc(y_test, test_proba)

        importance = dict(zip(feature_cols, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return TrainResult(
            model=self.model,
            feature_importance=importance,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained")

        df_clean = df.drop_nulls(subset=self.feature_cols)
        X = df_clean.select(self.feature_cols).to_numpy()

        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)[:, 1]

        result = df_clean.with_columns(
            pl.Series("ml_prediction", pred),
            pl.Series("ml_probability", proba),
        )

        return result

    def _calc_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

    def _calc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        sorted_indices = np.argsort(y_proba)[::-1]
        y_sorted = y_true[sorted_indices]

        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr = np.cumsum(y_sorted) / n_pos
        fpr = np.cumsum(1 - y_sorted) / n_neg

        auc = np.trapezoid(tpr, fpr)
        return float(auc)
