package schema_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"

	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func ExampleChoice() {
	team, err := schema.Choice("Which team should handle this ticket?",
		schema.Option{Label: "billing", Criterion: "payments, refunds, invoices"},
		schema.Option{Label: "technical", Criterion: "login, errors, outages"},
		schema.Option{Label: "other"}, // no description: renders as its label alone
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(team.Labels())
	for _, text := range team.RenderedOptions() {
		fmt.Println(text)
	}
	// Output:
	// [billing technical other]
	// billing: payments, refunds, invoices
	// technical: login, errors, outages
	// other
}

func ExampleScore() {
	urgency := schema.MustQuestion(schema.Score("How urgent is this?",
		"can wait", "this week", "today", map[string]any{"desc": "now", "sla_minutes": 15}))
	for i, text := range urgency.Legend() {
		fmt.Printf("%d: %s\n", i, text)
	}
	// Output:
	// 0: can wait
	// 1: this week
	// 2: today
	// 3: {"desc": "now", "sla_minutes": 15}
}

func ExampleNoulWith() {
	refund := schema.MustQuestion(schema.NoulWith("The customer asks for a refund.",
		schema.NoulSpec{TrueLabel: "refund", FalseLabel: "no refund"}))
	for _, text := range refund.RenderedOptions() { // false side first, always
		fmt.Println(text)
	}
	// Output:
	// no refund: no, the statement does not hold
	// refund: yes, the statement holds
}

func ExampleQuestion_WithOptionOrder() {
	q := schema.MustQuestion(schema.ChoiceOf("Which team?", "billing", "technical", "other"))
	shuffled, err := q.WithOptionOrder([]int{2, 0, 1}) // slot 0 shows option 2
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(shuffled.OptionOrder(), shuffled.Labels()) // answers still come back in label order
	_, err = q.WithOptionOrder([]int{0, 0, 1})
	fmt.Println(err)
	// Output:
	// [2 0 1] [billing technical other]
	// 'option_order' must be a permutation of range(3) -- one slot per option, each option once -- got [0, 0, 1]
}

func ExampleNewQuestions() {
	qs, err := schema.NewQuestions(
		schema.Ask("team", schema.MustQuestion(schema.ChoiceOf("Which team?", "billing", "technical"))),
		schema.Ask("refund", schema.MustQuestion(schema.Noul("The customer asks for a refund."))),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(qs.IDs())
	// Output: [team refund]
}

func ExampleParseQuestions() {
	qs, err := schema.ParseQuestions([]byte(`{
		"team": {"type": "choice", "instructions": "Which team?",
		         "criteria": {"billing": "payments", "technical": "login"}},
		"refund": {"type": "noul", "instructions": "The customer asks for a refund."}
	}`))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(qs.Len(), qs.IDs())

	// A refusal names the question and says what to fix, in laya's own words.
	_, err = schema.ParseQuestions([]byte(`{"team": {"type": "choice", "instructions": "Which team?"}}`))
	var refused *schema.QuestionError
	fmt.Println(errors.As(err, &refused), err)
	// Output:
	// 2 [team refund]
	// true question 'team': a choice question takes 'criteria' as a dict of label -> description, or a list of labels
}

func ExampleRenderCriterion() {
	text, _ := schema.RenderCriterion(map[string]any{"desc": "café", "max": 1e16})
	fmt.Println(text)
	// Output: {"desc": "café", "max": 1e+16}
}

func ExampleAnswer() {
	p := schema.Prediction{
		Model: schema.Model,
		Answers: schema.Answers{
			{ID: "team", Answer: &schema.ChoiceAnswer{
				Choice:        schema.Text("billing"),
				Probabilities: schema.Probabilities{{Label: schema.Text("billing"), P: 0.8123}, {Label: schema.Text("technical"), P: 0.1877}},
				Common:        schema.Common{Confidence: 0.3, AnswerConfidence: 0.8123, Action: schema.Action{ActProbability: 0.5}},
			}},
			{ID: "refund", Answer: &schema.NoulAnswer{Noul: 0.9, Common: schema.Common{Confidence: 0.9, AnswerConfidence: 0.9}}},
		},
		Usage: schema.Usage{InputTokens: 120, StateTokens: 40, StateTokensDropped: 12, Truncated: true, TruncatedQuestions: []string{"refund"}},
	}
	for _, a := range p.Answers {
		switch ans := a.Answer.(type) {
		case *schema.ChoiceAnswer:
			fmt.Println(a.ID, ans.Choice.String(), ans.AnswerConfidence)
		case *schema.ScoreAnswer:
			fmt.Println(a.ID, ans.Score)
		case *schema.NoulAnswer:
			fmt.Println(a.ID, ans.Noul)
		}
	}
	if p.Usage.Truncated {
		fmt.Println("state cut for", p.Usage.TruncatedQuestions)
	}
	out, err := json.Marshal(p) // laya's own result shape
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(out))
	// Output:
	// team billing 0.8123
	// refund 0.9
	// state cut for [refund]
	// {"model":"laya-rl-agent-onnx","answers":{"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.8123,"technical":0.1877},"confidence":0.3,"answer_confidence":0.8123,"action":{"act_probability":0.5}},"refund":{"type":"noul","noul":0.9,"confidence":0.9,"answer_confidence":0.9,"action":{"act_probability":0.0}}},"usage":{"input_tokens":120,"output_tokens":0,"state_tokens":40,"state_tokens_dropped":12,"truncated":true,"truncated_questions":["refund"]}}
}
