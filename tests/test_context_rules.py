import copy
import unittest

import numpy as np

from semantic_pipeline import (
    CONTEXT_RULES,
    LABEL_PATTERN_SETS,
    build_context_adjustment,
    load_context_rules_config,
    validate_context_rules_config,
)


class ContextRulesConfigTest(unittest.TestCase):
    def test_default_rules_load_and_validate(self):
        config = load_context_rules_config()
        validate_context_rules_config(config)
        self.assertIn("traffic", config["label_pattern_sets"])
        self.assertGreaterEqual(len(config["context_rules"]), 1)

    def test_default_json_matches_module_exports(self):
        config = load_context_rules_config()
        self.assertEqual(config["label_pattern_sets"], LABEL_PATTERN_SETS)
        self.assertEqual(config["context_rules"], CONTEXT_RULES)

    def test_invalid_group_reference_is_rejected(self):
        config = load_context_rules_config()
        bad = copy.deepcopy(config)
        bad["context_rules"][0]["boost"]["missing_group"] = 1.2
        with self.assertRaisesRegex(ValueError, "unknown boost group"):
            validate_context_rules_config(bad)

    def test_malformed_factor_is_rejected(self):
        config = load_context_rules_config()
        bad = copy.deepcopy(config)
        bad["context_rules"][0]["suppress"] = {"construction": "not-a-number"}
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            validate_context_rules_config(bad)

    def test_explicit_json_config_matches_default_adjustment(self):
        labels = ["Speech", "Rain", "Car horn", "Bird", "Inside, small room"]
        context = {"scene": {"indoor_outdoor": "outdoor"}, "weather": {"condition": "rain"}}
        config = load_context_rules_config()
        default_weights, default_impacts, default_signals = build_context_adjustment(labels, context)
        json_weights, json_impacts, json_signals = build_context_adjustment(labels, context, rules_config=config)
        np.testing.assert_allclose(default_weights, json_weights)
        self.assertEqual([impact.__dict__ for impact in default_impacts], [impact.__dict__ for impact in json_impacts])
        self.assertEqual([signal.__dict__ for signal in default_signals], [signal.__dict__ for signal in json_signals])


if __name__ == "__main__":
    unittest.main()
