import pandas as pd

from fedrec.config import RetrievalConfig
from fedrec.data import MOVIE_GENRES
from fedrec.retrieval import HybridRetriever, build_candidate_universe


def make_items(count: int = 20) -> pd.DataFrame:
    rows = []
    for item_id in range(1, count + 1):
        row = {genre: 0 for genre in MOVIE_GENRES}
        row.update(
            {
                "item_id": item_id,
                "item_title": f"Movie {item_id}",
                "release_year": 1990,
                "Action": int(item_id % 2 == 0),
                "Drama": int(item_id % 2 == 1),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_controlled_pool_retains_targets_only_at_pool_level():
    config = RetrievalConfig(mother_pool_size=8, recall_size=4)
    pool = build_candidate_universe(
        user_id=7,
        catalog_items=range(1, 21),
        history_items={1, 2, 3},
        target_items={4, 5},
        other_explicit_items={6, 7},
        config=config,
        seed=2026,
    )
    assert len(pool) == 8
    assert {4, 5}.issubset(pool)
    assert not {1, 2, 3, 6, 7}.intersection(pool)


def test_hybrid_retriever_returns_unique_ranked_items():
    items = make_items()
    history = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "item_id": [1, 2, 3, 2, 3, 4],
            "label": [1, 1, 0, 1, 0, 1],
            "timestamp": [1, 2, 3, 1, 2, 3],
        }
    )
    retriever = HybridRetriever(
        history,
        items,
        RetrievalConfig(mother_pool_size=10, recall_size=5),
    )
    recalled = retriever.retrieve(1, range(1, 21))
    assert len(recalled) == 5
    assert recalled["item_id"].is_unique
    assert recalled["retrieval_rank"].tolist() == [1, 2, 3, 4, 5]
    assert not set(recalled["item_id"]).intersection({1, 2, 3})

