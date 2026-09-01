import pandas as pd

from fedrec.config import DataConfig
from fedrec.data import fixed_holdout_split


def test_fixed_holdout_preserves_positive_and_negative_labels():
    interactions = pd.DataFrame(
        {
            "user_id": [1] * 30,
            "item_id": list(range(1, 31)),
            "rating": [5.0] * 20 + [2.0] * 10,
            "timestamp": list(range(30)),
            "label": [1] * 20 + [0] * 10,
        }
    )
    config = DataConfig(
        min_user_interactions=20,
        validation_holdout_size=5,
        test_holdout_size=10,
    )
    output = fixed_holdout_split(interactions, config)
    assert output["split"].value_counts().to_dict() == {
        "train": 15,
        "test": 10,
        "validation": 5,
    }
    for split in ("train", "validation", "test"):
        assert set(output.loc[output["split"].eq(split), "label"]) == {0, 1}

