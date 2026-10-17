"""Guard-matrix expansion: pin evaluate_shortlist's parameter validators."""
import sys
sys.path.insert(0, "/Users/aashish/github contribution /GIT/laya")

from laya.evals import Dataset, EvalError, Example
from laya.evals_shortlist import evaluate_shortlist


QUESTION = {"intent": {"type": "choice", "instructions": "Which?",
                       "criteria": ["a", "b", "c"]}}


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
    FAIL = []
    
    # --- k parameter guards (line 118) ---
    agent = ChoiceAgent({})
    dataset = Dataset([sample("one", "a")])
    embed = TableEmbed()
    
    check("k/positive_int_ok", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), None)
    check("k/zero_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=0, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    check("k/negative_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=-1, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    check("k/bool_true_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=True, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    check("k/bool_false_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=False, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    check("k/string_raises", lambda: evaluate_shortlist(agent, dataset, embed, k="2", checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    # --- embed_fn parameter guards (line 120-121) ---
    check("embed_fn/callable_ok", lambda: evaluate_shortlist(agent, dataset, lambda x: [[1.0, 0.0] for _ in x], k=2, checkpoint_id="ck@1", embedder_id="emb@1"), None)
    check("embed_fn/none_raises", lambda: evaluate_shortlist(agent, dataset, None, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    check("embed_fn/int_raises", lambda: evaluate_shortlist(agent, dataset, 123, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    # --- checkpoint_id guards (line 122-124) ---
    check("checkpoint_id/nonempty_string_ok", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), None)
    check("checkpoint_id/empty_string_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="", embedder_id="emb@1"), EvalError)
    check("checkpoint_id/whitespace_only_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="   ", embedder_id="emb@1"), EvalError)
    check("checkpoint_id/int_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id=123, embedder_id="emb@1"), EvalError)
    
    # --- embedder_id guards (same pattern as checkpoint_id) ---
    check("embedder_id/nonempty_string_ok", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), None)
    check("embedder_id/empty_string_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="ck@1", embedder_id=""), EvalError)
    check("embedder_id/whitespace_only_raises", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="ck@1", embedder_id="   "), EvalError)
    
    # --- dataset validation (lines 32-45) ---
    check("dataset/non_empty_ok", lambda: evaluate_shortlist(agent, dataset, embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), None)
    check("dataset/empty_raises", lambda: evaluate_shortlist(agent, Dataset([]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    check("dataset/list_instead_of_object_raises", lambda: evaluate_shortlist(agent, [], embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    # Malformed dataset examples
    ex_no_expected = type('Example', (), {'state':'x','questions':{},'expected':None})()
    check("dataset/missing_expected_raises", lambda: evaluate_shortlist(agent, Dataset([ex_no_expected]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    ex_wrong_q_type = type('Example', (), {'state':'x','questions':{'q':{'type':'text'}},'expected':{'q':'a'}})()
    check("dataset/wrong_question_type_raises", lambda: evaluate_shortlist(agent, Dataset([ex_wrong_q_type]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    ex_bad_criteria = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':123}},'expected':{'q':'a'}})()
    check("dataset/bad_criteria_type_raises", lambda: evaluate_shortlist(agent, Dataset([ex_bad_criteria]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    ex_empty_criteria = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':[]}},'expected':{'q':'a'}})()
    check("dataset/bad_criteria_content_raises", lambda: evaluate_shortlist(agent, Dataset([ex_empty_criteria]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    ex_dup_labels = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':['a','a']}},'expected':{'q':'a'}})()
    check("dataset/duplicate_labels_raises", lambda: evaluate_shortlist(agent, Dataset([ex_dup_labels]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    ex_nonstring_label = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':[123]}},'expected':{'q':'a'}})()
    check("dataset/nonstring_label_raises", lambda: evaluate_shortlist(agent, Dataset([ex_nonstring_label]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    ex_expected_outside = type('Example', (), {'state':'x','questions':{'q':{'type':'choice','criteria':['a']}},'expected':{'q':'b'}})()
    check("dataset/expected_not_in_criteria_raises", lambda: evaluate_shortlist(agent, Dataset([ex_expected_outside]), embed, k=2, checkpoint_id="ck@1", embedder_id="emb@1"), EvalError)
    
    print("--- summary ---")
    passed = sum(1 for l in open('/dev/stdin') if 'PASS' in l)
    failed = sum(1 for l in open('/dev/stdin') if 'FAIL' in l)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
