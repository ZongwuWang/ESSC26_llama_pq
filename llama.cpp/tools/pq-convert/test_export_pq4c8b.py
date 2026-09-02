#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_pq4c8b import (
    llama_rope_row_permutation,
    load_attention_head_counts,
    permute_output_axis,
)


class ExportPQ4C8BTest(unittest.TestCase):
    def test_rope_permutation_matches_llama_converter_layout(self):
        n_head = 2
        head_dim = 8
        n_out = n_head * head_dim
        expected = (
            np.arange(n_out)
            .reshape(n_head, 2, head_dim // 2)
            .swapaxes(1, 2)
            .reshape(-1)
        )
        np.testing.assert_array_equal(
            llama_rope_row_permutation(n_out, n_head), expected
        )

    def test_output_axis_permutation_updates_codes_and_scales(self):
        permutation = llama_rope_row_permutation(8, 2)
        codes = np.arange(2 * 8 * 3).reshape(2, 8, 3)
        scales = np.arange(8)
        np.testing.assert_array_equal(
            permute_output_axis(codes, permutation, axis=1),
            codes[:, permutation, :],
        )
        np.testing.assert_array_equal(
            permute_output_axis(scales, permutation, axis=0),
            scales[permutation],
        )

    def test_model_config_preserves_distinct_gqa_head_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(
                json.dumps({"num_attention_heads": 32, "num_key_value_heads": 8}),
                encoding="utf-8",
            )
            self.assertEqual(load_attention_head_counts(config), (32, 8))

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            llama_rope_row_permutation(10, 3)
        with self.assertRaises(ValueError):
            permute_output_axis(np.zeros((2, 3)), np.arange(4), axis=1)


if __name__ == "__main__":
    unittest.main()
