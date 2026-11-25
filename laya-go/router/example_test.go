package router_test

import (
	"context"
	"fmt"

	"github.com/NandhaKishorM/laya/laya-go/agent"
	"github.com/NandhaKishorM/laya/laya-go/router"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

func ExampleRouter_Route() {
	r, _ := router.New()
	var qs schema.Questions

	for _, state := range []string{
		"Please refund the duplicate charge from March.",
		"मुझसे दो बार शुल्क लिया गया",
		"",
	} {
		d, _ := r.Route(state, qs)
		fmt.Println(d.Model, "|", d.Reason)
	}
	// An explicit model wins over everything the text says.
	d, _ := r.Route("मुझसे दो बार शुल्क लिया गया", qs, router.ForModel("en"))
	fmt.Println(d)
	// Output:
	// english | English Latin text
	// multilingual | non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it
	// multilingual | no letters detected in state; using default (multilingual)
	// RouteDecision(model='english', reason="explicit model='en'")
}

func ExampleRouter_RouteBatch() {
	r, _ := router.New(router.WithDefault("english"))
	decisions, _ := r.RouteBatch([]router.Request{
		{State: "12345"},
		{State: "Kunden wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung"},
		{State: "anything", Options: []router.CallOption{router.ForLang("de_DE.UTF-8")}},
	})
	for _, d := range decisions {
		fmt.Println(d.Model)
	}
	// Output:
	// english
	// multilingual
	// multilingual
}

func ExampleEnglishFromCode() {
	for _, code := range []string{"en_US.UTF-8", "pt-BR", "C.UTF-8", ""} {
		english, ok := router.EnglishFromCode(code)
		fmt.Printf("%q english=%v known=%v\n", code, english, ok)
	}
	// Output:
	// "en_US.UTF-8" english=true known=true
	// "pt-BR" english=false known=true
	// "C.UTF-8" english=false known=false
	// "" english=false known=false
}

func ExampleNormaliseName() {
	c, _ := router.NormaliseName("  Typed_Decisions ")
	fmt.Println(c)
	_, err := router.NormaliseName("gpt")
	fmt.Println(err)
	// Output:
	// typed-decisions
	// unknown model 'gpt'; choose one of ['english', 'multilingual', 'typed-decisions'] (or an alias: ['decisions', 'default', 'en', 'laya', 'laya-multilingual', 'laya-typed-decisions', 'ml', 'multi', 'typed', 'typed_decisions'])
}

func ExampleRouter_AddHook() {
	r, _ := router.New()
	// A hook may replace the decision; Route returns the replacement.
	_ = r.AddHook(&router.HookFuncs{OnRouteFunc: func(c *router.RouteContext) {
		fmt.Println("saw", c.Decision.Model)
		if c.Decision.Model == router.English {
			c.Decision = router.Decision{Model: router.Multilingual, Repo: c.Decision.Repo, Reason: "pinned by policy"}
		}
	}})
	d, _ := r.Route("Please refund the duplicate charge from March.", schema.Questions{})
	fmt.Println(d.Model, "|", d.Reason)
	// Output:
	// saw english
	// multilingual | pinned by policy
}

func ExampleRouter_Lease() {
	// The factory owns where a checkpoint comes from; a real one opens the weights with
	// agent.Open and an onnx session. This one stands in.
	var open router.AgentFactory = func(ctx context.Context, c router.Checkpoint) (*agent.Agent, error) {
		return nil, fmt.Errorf("no weights for %s in this example", c)
	}
	r, _ := router.New(router.WithAgentFactory(open), router.WithMaxLoaded(1))
	defer r.Close()

	_, err := r.Lease(context.Background(), "english")
	fmt.Println(err)
	fmt.Println(r.Loaded(), r.MaxLoaded())
	// Output:
	// router: cannot load the english checkpoint: no weights for english in this example
	// [] 1
}

func ExampleRouter_ExpectedSHA256() {
	t := "model.safetensors"
	r, _ := router.New(router.WithSHA256Digests(map[string]map[string]string{
		"english":      {t: "aaaa"},
		"multilingual": nil, // pin nothing, even where LAYA_SHA256_DIGESTS names files
	}))
	// A factory asks for each checkpoint's map and hands a non-nil one to agent.WithExpectedSHA256.
	shared := map[string]string{"tokenizer/tokenizer.json": "dddd", t: "cccc"}
	for _, name := range []string{"english", "multilingual", "typed-decisions"} {
		m, _ := r.ExpectedSHA256(name, shared)
		fmt.Println(name, m)
	}
	m, _ := r.ExpectedSHA256("typed-decisions", nil)
	fmt.Println(m == nil)
	// Output:
	// english map[model.safetensors:aaaa tokenizer/tokenizer.json:dddd]
	// multilingual map[model.safetensors:cccc tokenizer/tokenizer.json:dddd]
	// typed-decisions map[model.safetensors:cccc tokenizer/tokenizer.json:dddd]
	// true
}
