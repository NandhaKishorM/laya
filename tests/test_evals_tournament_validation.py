"""Guard-matrix expansion: pin evaluate_tournament's parameter validators."""
import sys
sys.path.insert(0, "/Users/aashish/github contribution /GIT/laya")

from laya.evals import Dataset, EvalError, Example
from laya.evals_tournament import evaluate_tournament


QUESTION = {"intent": {"type": "choice", "instructions": "Which?",
                       "criteria": ["a", "b", "c", "d", "e"]}}


class TableEmbed:
    def __call__(self, texts):
        return [[1.0, 0.0] for _ in texts]


class ChoiceAgent:
    def __init__(self, chosen):
        self.chosen = chosen

    def system_one(self, state, questions):
        return {"model": "stub", "answers": {"intent": {
            "type": "choice", "choice": self.chosen.get(state, "a"),
            "answer_confidence": 0.7,
        }}}


def sample(state, gold):
    return Example(state, QUESTION, {"intent": gold}, tags=("ok",))


def check(name, fn, want_error=None, want_attr=None):
    try:
        result = fn()
        if want_error:
            print(f"FAIL {name}: expected {want_error.__name__} but got {result}")
            return False
        elif want_attr is not None and result != want_attr:
            print(f"FAIL {name}: expected {want_attr!r} but got {result!r}")
            return False
        else:
            print(f"PASS {name}")
            return True
    except Exception as e:
        if want_error and isinstance(e, want_error):
            print(f"PASS {name} ({type(e).__name__})")
            return True
        else:
            print(f"FAIL {name}: unexpected {type(e).__name__}: {e}")
            return False


def main():
    agent = ChoiceAgent({})
    dataset = Dataset([sample("one", "a")])
    embed = TableEmbed()
    
    # --- group_size guards (line 110) ---
    check("group_size/min_2_ok", lambda: evaluate_tournament(agent, dataset, group_size=2, checkpoint_id="ck@1"), None)
    check("group_size/3_ok", lambda: evaluate_tournament(agent, dataset, group_size=3, checkpoint_id="ck@1"), None)
    check("group_size/zero_raises", lambda: evaluate_tournament(agent, dataset, group_size=0, checkpoint_id="ck@1"), EvalError)
    check("group_size/negative_raises", lambda: evaluate_tournament(agent, dataset, group_size=-1, checkpoint_id="ck@1"), EvalError)
    check("group_size/bool_true_raises", lambda: evaluate_tournament(agent, dataset, group_size=True, checkpoint_id="ck@1"), EvalError)
    check("group_size/bool_false_raises", lambda: evaluate_tournament(agent, dataset, group_size=False, checkpoint_id="ck@1"), EvalError)
    check("group_size/string_raises", lambda: evaluate_tournament(agent, dataset, group_size="2", checkpoint_id="ck@1"), EvalError)
    
    # --- checkpoint_id guards (line 112-113) ---
    check("checkpoint_id/nonempty_string_ok", lambda: evaluate_tournament(agent, dataset, group_size=3, checkpoint_id="ck@1"), None)
    check("checkpoint_id/empty_string_raises", lambda: evaluate_tournament(agent, dataset, group_size=3, checkpoint_id=""), EvalError)
    check("checkpoint_id/whitespace_only_raises", lambda: evaluate_tournament(agent, dataset, group_size=3, checkpoint_id="   "), EvalError)
    check("checkpoint_id/int_raises", lambda: evaluate_tournament(agent, dataset, group_size=3, checkpoint_id=123), EvalError)
    
    # --- dataset validation (lines 27-40) ---
    check("dataset/non_empty_ok", lambda: evaluate_tournament(agent, dataset, group_size=3, checkpoint_id="ck@1"), None)
    check("dataset/empty_raises", lambda: evaluate_tournament(agent, Dataset([]), group_size=3, checkpoint_id="ck@1"), EvalError)
    check("dataset/list_instead_of_object_raises", lambda: evaluate_tournament(agent, [], group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_no_expected = type('Example', (), {'state':'x','questions':{},'expected':None})()
    check("dataset/missing_expected_raises", lambda: evaluate_tournament(agent, Dataset([ex_no_expected]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_wrong_q_type = type('Example', (), {'state':'x','questions':{'q':{'type':'text'}},'expected':{'q':'a'}})()
    check("dataset/wrong_question_type_raises", lambda: evaluate_tournament(agent, Dataset([ex_wrong_q_type]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_bad_criteria = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':123}},'expected':{'q':'a'}})()
    check("dataset/bad_criteria_type_raises", lambda: evaluate_tournament(agent, Dataset([ex_bad_criteria]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_empty_criteria = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':[]}},'expected':{'q':'a'}})()
    check("dataset/bad_criteria_content_raises", lambda: evaluate_tournament(agent, Dataset([ex_empty_criteria]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_dup_labels = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':['a','a']}},'expected':{'q':'a'}})()
    check("dataset/duplicate_labels_raises", lambda: evaluate_tournament(agent, Dataset([ex_dup_labels]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_nonstring_label = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':[123]}},'expected':{'q':'a'}})()
    check("dataset/nonstring_label_raises", lambda: evaluate_tournament(agent, Dataset([ex_nonstring_label]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    ex_expected_outside = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':['a']}},'expected':{'q':'b'}})()
    check("dataset/expected_not_in_criteria_raises", lambda: evaluate_tournament(agent, Dataset([ex_expected_outside]), group_size=3, checkpoint_id="ck@1"), EvalError)
    
    print("--- summary ---")
    passed = sum(1 for l in open('/dev/stdin') if 'PASS' in l)
    failed = sum(1 for l in open('/dev/stdin') if 'FAIL' in l)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
