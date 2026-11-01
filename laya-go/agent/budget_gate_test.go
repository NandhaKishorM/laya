package agent_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func budgetState() string { return strings.Repeat("please refund my order ", 20) }

func TestWithMaxLenAndHeadMaxLenReachTheSession(t *testing.T) {
	a, s := open(t, `{}`)
	defer a.Close()
	qs := oneChoice(t)
	ctx := context.Background()
	if _, err := a.Predict(ctx, budgetState(), qs); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Predict(ctx, budgetState(), qs, agent.WithMaxLen(64)); err != nil {
		t.Fatal(err)
	}
	if len(s.runs) != 2 || s.runs[1].Length > 64 || s.runs[0].Length <= 64 {
		t.Fatalf("session lengths %+v: WithMaxLen(64) did not cap the sequence", s.runs)
	}
	short := "refund it"
	n := len(s.runs)
	for _, head := range []int{56, 8} {
		if _, err := a.Predict(ctx, short, qs, agent.WithMaxLen(64), agent.WithHeadMaxLen(head)); err != nil {
			t.Fatal(err)
		}
	}
	if wide, narrow := s.runs[n].Length, s.runs[n+1].Length; narrow >= wide {
		t.Errorf("WithHeadMaxLen did not move the question budget: lengths %d (56) and %d (8)", wide, narrow)
	}
}

// The start hook's context starts from the call's budgets, so a hook sees what the caller asked.
func TestPredictContextIsSeededWithTheBudgets(t *testing.T) {
	a, _ := open(t, `{}`)
	defer a.Close()
	var seen [2]int
	h := &agent.HookFuncs{OnStart: func(c *agent.PredictContext) { seen = [2]int{c.MaxLen, c.HeadMaxLen} }}
	if _, err := a.Predict(context.Background(), "refund", oneChoice(t), agent.WithHooks(h), agent.WithMaxLen(70), agent.WithHeadMaxLen(30)); err != nil {
		t.Fatal(err)
	}
	if seen != [2]int{70, 30} {
		t.Errorf("hook saw MaxLen/HeadMaxLen %v, want [70 30]", seen)
	}
}

func TestPredictLongRefusesBudgets(t *testing.T) {
	a, _ := open(t, `{}`)
	defer a.Close()
	for name, opt := range map[string]agent.CallOption{"max": agent.WithMaxLen(12), "head": agent.WithHeadMaxLen(12)} {
		if _, err := a.PredictLong(context.Background(), "refund", oneChoice(t), opt); err == nil {
			t.Errorf("%s accepted by PredictLong", name)
		}
	}
	if _, err := a.PredictLong(context.Background(), "refund", oneChoice(t), agent.WithMaxLen(0)); err != nil {
		t.Errorf("WithMaxLen(0) is no override: %v", err)
	}
}

func TestHasLangTemperatures(t *testing.T) {
	plain, _ := open(t, `{}`)
	defer plain.Close()
	if plain.HasLangTemperatures() {
		t.Error("an agent with no language table reports one")
	}
	a, err := agent.Open(toyCheckpoint(`{}`), &fakeSession{},
		agent.WithLangTemperatures(map[string]map[string]any{"de": {"temperature": []any{2.0, 2.0, 2.0}}}))
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if !a.HasLangTemperatures() {
		t.Error("an agent with a language table reports none")
	}
}

func lowConfidence() []schema.Prediction {
	return []schema.Prediction{{Answers: schema.Answers{{ID: "q", Answer: &schema.ChoiceAnswer{
		Common: schema.Common{Confidence: 0.4, AnswerConfidence: 0.4}}}}}}
}

func TestNewGate(t *testing.T) {
	var mc *agent.MinConfidenceError
	if _, err := agent.NewGate(agent.WithMinConfidence(2)); !errors.As(err, &mc) {
		t.Fatalf("invalid threshold: %v", err)
	}
	preds := lowConfidence()
	g, err := agent.NewGate(agent.WithMinConfidence(0.9))
	if err != nil {
		t.Fatal(err)
	}
	g.Apply(preds)
	c := &preds[0].Answers[0].Answer.(*schema.ChoiceAnswer).Common
	if !c.LowConfidence || c.Abstention != "abstained" || c.AbstentionThreshold == nil {
		t.Errorf("gate did not act: %+v", c)
	}
	preds = lowConfidence()
	none, _ := agent.NewGate()
	none.Apply(preds)
	(*agent.Gate)(nil).Apply(preds)
	if c := preds[0].Answers[0].Answer.(*schema.ChoiceAnswer).Common; c.Abstention != "" {
		t.Errorf("an empty gate acted: %+v", c)
	}
}

func oneChoice(t *testing.T) schema.Questions {
	t.Helper()
	qs, err := schema.NewQuestions(schema.Ask("intent", schema.MustQuestion(schema.ChoiceOf("What?", "status", "refund"))))
	if err != nil {
		t.Fatal(err)
	}
	return qs
}
