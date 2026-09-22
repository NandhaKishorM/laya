import unittest
from laya.agent import Agent
from laya.common import render_options

class CriteriaTests(unittest.TestCase):
    def test_noul_criteria_boolean_keys(self):
        qdef = {
            "type": "noul",
            "instructions": "Is this a refund?",
            "criteria": {True: "User requests money back", False: "User does not ask for money back"}
        }
        internal = Agent._to_internal(qdef)
        self.assertIn("true", internal["crit"])
        self.assertIn("false", internal["crit"])
        opts = render_options(internal)
        self.assertEqual(len(opts), 2)
        self.assertTrue(opts[0].startswith("false: User does not ask for money back"))
        self.assertTrue(opts[1].startswith("true: User requests money back"))

    def test_choice_criteria_list_expansion(self):
        qdef = {
            "type": "choice",
            "instructions": "Select urgency",
            "criteria": ["low", "medium", "high"]
        }
        internal = Agent._to_internal(qdef)
        self.assertEqual(list(internal["crit"].keys()), ["low", "medium", "high"])

if __name__ == "__main__":
    unittest.main()
