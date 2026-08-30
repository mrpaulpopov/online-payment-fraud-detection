from typing import Any

import lightgbm as lgb


def lightgbm_model(
        train_data: lgb.Dataset,
        valid_data: lgb.Dataset,
        params: dict[str, Any]) -> lgb.Booster:
    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,  # 1000-3000
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100)]
    )
    return model
