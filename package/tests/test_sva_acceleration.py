import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sva_acceleration import (
    HalvingRound,
    SparseCFGCache,
    parse_halving_schedule,
    select_topk,
)


class AccelerationHelperTests(unittest.TestCase):
    def test_parse_halving_schedule(self):
        self.assertEqual(
            parse_halving_schedule("15:4,30:2,45:1"),
            [HalvingRound(15, 4), HalvingRound(30, 2), HalvingRound(45, 1)],
        )

    def test_topk_is_score_first_seed_tiebreak(self):
        scores = {0: 0.2, 1: 0.9, 2: 0.9, 3: 0.1}
        self.assertEqual(select_topk([0, 1, 2, 3], scores, 2), [1, 2])

    def test_sparse_cfg_cache_select(self):
        cache = SparseCFGCache(
            unconditional=[torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
            last_refresh_step=7,
        )
        selected = cache.select([2, 0])
        self.assertEqual(selected.last_refresh_step, 7)
        self.assertEqual([x.item() for x in selected.unconditional], [3.0, 1.0])


if __name__ == "__main__":
    unittest.main()
