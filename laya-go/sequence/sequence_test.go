package sequence_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/lang"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// Python's literals, which the package does not export.
const (
	optionTokenCap       = 48
	minInstructionTokens = 8
)

func toy(tb testing.TB) *tokenizer.Tokenizer {
	tb.Helper()
	spec, config := fixtures.ToyTokenizer()
	tok, err := tokenizer.Parse(spec, config)
	if err != nil {
		tb.Fatal(err)
	}
	return tok
}

func ids(tok *tokenizer.Tokenizer, text string) []int32 { return tok.Encode(text) }

func TestSerializeState(t *testing.T) {
	type named string
	cases := []struct {
		name  string
		state any
		want  string
	}{
		{"string passes through", "plain {text}", "plain {text}"},
		{"named string", named("x"), "x"},
		{"nil is null", nil, "null"},
		{"bool", true, "true"},
		{"int", 7, "7"},
		{"float spelled as Python", 1e16, "1e+16"},
		{"whole float", 2.0, "2.0"},
		{"negative zero", math.Copysign(0, -1), "-0.0"},
		{"list", []any{1, "a", nil, 1.5}, `[1, "a", null, 1.5]`},
		{"string slice", []string{"a", "b"}, `["a", "b"]`},
		{"map in sorted order", map[string]any{"b": 1, "a": 2}, `{"a": 2, "b": 1}`},
		{"ordered object keeps order", lang.Object{{Key: "z", Value: 1}, {Key: "a", Value: 2}}, `{"z": 1, "a": 2}`},
		{"non-ASCII is not escaped", map[string]any{"k": "é中"}, `{"k": "é中"}`},
		{"nested typed slice", map[string][]int{"n": {1, 2}}, `{"n": [1, 2]}`},
		{"pointer is followed", new(int), "0"},
		{"nil slice is an empty list", []any(nil), "[]"},
		{"json.Number keeps its type", []any{json.Number("2"), json.Number("2.0"), json.Number("1e16")}, "[2, 2.0, 1e+16]"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := sequence.SerializeState(c.state)
			if err != nil || got != c.want {
				t.Errorf("SerializeState = %q, %v; want %q", got, err, c.want)
			}
		})
	}
}

func TestSerializeStateRefuses(t *testing.T) {
	type point struct{ X int }
	cases := map[string]any{
		"struct":            point{1},
		"channel":           make(chan int),
		"function":          func() {},
		"bytes":             []byte("raw"),
		"int-keyed map":     map[int]string{1: "a"},
		"invalid UTF-8":     []any{"\xff"},
		"struct in a list":  []any{point{1}},
		"struct in a map":   map[string]any{"k": point{}},
		"struct behind ptr": &point{1},
	}
	for name, state := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := sequence.SerializeState(state); !errors.Is(err, sequence.ErrUnsupportedState) {
				t.Errorf("err = %v, want ErrUnsupportedState", err)
			}
		})
	}
}

func TestSerializeStateCycle(t *testing.T) {
	loop := []any{nil}
	loop[0] = loop
	if _, err := sequence.SerializeState(loop); !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Errorf("a self-containing list: err = %v, want ErrUnsupportedState", err)
	}
}

func TestEncodeState(t *testing.T) {
	tok := toy(t)
	cases := []struct {
		name  string
		state any
		text  string
		keep  sequence.Keep
	}{
		{"string keeps the start", "the in", "the in", sequence.KeepStart},
		{"list keeps the end", []any{"a", "b"}, `["a", "b"]`, sequence.KeepEnd},
		{"string slice keeps the end", []string{"a"}, `["a"]`, sequence.KeepEnd},
		{"array keeps the end", [2]int{1, 2}, "[1, 2]", sequence.KeepEnd},
		{"empty list keeps the end", []any{}, "[]", sequence.KeepEnd},
		{"map keeps the start", map[string]any{"a": 1}, `{"a": 1}`, sequence.KeepStart},
		{"ordered object keeps the start", lang.Object{{Key: "a", Value: 1}}, `{"a": 1}`, sequence.KeepStart},
		{"the mask token is scrubbed", "x [MASK] y[MASK]", "x   y ", sequence.KeepStart},
		{"the mask token inside a list is scrubbed", []any{"[MASK]"}, `[" "]`, sequence.KeepEnd},
		{"nil", nil, "null", sequence.KeepStart},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			st, err := sequence.EncodeState(tok, c.state)
			if err != nil {
				t.Fatal(err)
			}
			if want := ids(tok, c.text); !slices.Equal(st.IDs, want) {
				t.Errorf("ids = %v, want %v (%q)", st.IDs, want, c.text)
			}
			if st.Keep != c.keep {
				t.Errorf("Keep = %q, want %q", st.Keep, c.keep)
			}
			if slices.Contains(st.IDs, fixtures.ToyMask) && !strings.Contains(c.text, "[MASK]") {
				t.Errorf("a mask id reached the state: %v", st.IDs)
			}
		})
	}
	if st, err := sequence.EncodeState(tok, ""); err != nil || len(st.IDs) != 0 {
		t.Errorf("empty state = %v, %v", st, err)
	}
}

func TestEncodeStateRefuses(t *testing.T) {
	if _, err := sequence.EncodeState(toy(t), make(chan int)); !errors.Is(err, sequence.ErrUnsupportedState) {
		t.Errorf("unsupported state: %v", err)
	}
	if _, err := sequence.EncodeState(nil, "x"); err == nil {
		t.Error("a nil tokenizer was accepted")
	}
	for _, role := range []string{"cls_token", "sep_token", "mask_token"} {
		spec, config := fixtures.ToyTokenizer()
		without := strings.Replace(string(config), `"`+role+`":"[`, `"x_`+role+`":"[`, 1)
		tok, err := tokenizer.Parse(spec, []byte(without))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := sequence.EncodeState(tok, "x"); !errors.Is(err, sequence.ErrMissingSpecialToken) {
			t.Errorf("no %s: err = %v, want ErrMissingSpecialToken", role, err)
		}
		q := schema.MustQuestion(schema.Noul("holds"))
		if _, err := sequence.BuildHead(tok, q, 64); !errors.Is(err, sequence.ErrMissingSpecialToken) {
			t.Errorf("BuildHead without %s: err = %v", role, err)
		}
	}
}

func TestBuildHead(t *testing.T) {
	tok := toy(t)
	q := schema.MustQuestion(schema.Choice("th", schema.Option{Label: "a", Criterion: "in"}, schema.Option{Label: "b"}))
	head, err := sequence.BuildHead(tok, q, 192)
	if err != nil {
		t.Fatal(err)
	}
	want := []int32{fixtures.ToyCLS}
	want = append(want, ids(tok, "choice question: th")...)
	want = append(want, fixtures.ToySEP)
	m0 := len(want)
	want = append(want, fixtures.ToyMask)
	want = append(want, ids(tok, " a: in")...)
	m1 := len(want)
	want = append(want, fixtures.ToyMask)
	want = append(want, ids(tok, " b")...)
	want = append(want, fixtures.ToySEP)
	if !slices.Equal(head.IDs, want) {
		t.Errorf("head = %v\nwant   %v", head.IDs, want)
	}
	if !slices.Equal(head.Markers, []int{m0, m1}) {
		t.Errorf("markers = %v, want %v", head.Markers, []int{m0, m1})
	}
	if head.Stats != (sequence.Stats{Options: 2, OptionsDistinct: 2}) {
		t.Errorf("stats = %+v", head.Stats)
	}
}

func TestBuildHeadOrderAndScrub(t *testing.T) {
	tok := toy(t)
	q := schema.MustQuestion(schema.Choice("pick [MASK] now", schema.Option{Label: "a", Criterion: "x[MASK]y"}, schema.Option{Label: "b"}))
	ordered, err := q.WithOptionOrder([]int{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	head, err := sequence.BuildHead(tok, ordered, 192)
	if err != nil {
		t.Fatal(err)
	}
	if n := countID(head.IDs, fixtures.ToyMask); n != 2 {
		t.Errorf("%d mask ids in the head, want exactly the 2 option markers: %v", n, head.IDs)
	}
	// slot 0 shows option 1 ("b"), slot 1 shows option 0 ("a: x y")
	if got, want := head.IDs[head.Markers[0]+1:head.Markers[1]], ids(tok, " b"); !slices.Equal(got, want) {
		t.Errorf("slot 0 = %v, want option b %v", got, want)
	}
	tail := head.IDs[head.Markers[1]+1 : len(head.IDs)-1]
	if want := ids(tok, " a: x y"); !slices.Equal(tail, want) {
		t.Errorf("slot 1 = %v, want %v", tail, want)
	}
}

func countID(s []int32, id int32) (n int) {
	for _, v := range s {
		if v == id {
			n++
		}
	}
	return n
}

func TestBuildHeadBudgets(t *testing.T) {
	tok := toy(t)
	long := strings.Repeat("w", 200)
	t.Run("each option is cut to 48 tokens after its mask", func(t *testing.T) {
		q := schema.MustQuestion(schema.ChoiceOf("p", "a", long))
		head, err := sequence.BuildHead(tok, q, 4096)
		if err != nil {
			t.Fatal(err)
		}
		if span := head.IDs[head.Markers[1] : len(head.IDs)-1]; len(span) != 1+optionTokenCap {
			t.Errorf("long option span = %d ids, want %d", len(span), 1+optionTokenCap)
		}
		if head.Stats.TokensPerOption != nil {
			t.Errorf("no cap was needed, got %d", *head.Stats.TokensPerOption)
		}
	})
	t.Run("options leaving under 16 are re-capped, floor 4", func(t *testing.T) {
		labels := make([]string, 30)
		for i := range labels {
			labels[i] = long + string(rune('a'+i%26)) + string(rune('a'+i/26))
		}
		q := schema.MustQuestion(schema.ChoiceOf("an instruction that is long enough to be cut", labels...))
		head, err := sequence.BuildHead(tok, q, 64)
		if err != nil {
			t.Fatal(err)
		}
		if head.Stats.TokensPerOption == nil || *head.Stats.TokensPerOption != 4 {
			t.Fatalf("TokensPerOption = %v, want the floor 4", head.Stats.TokensPerOption)
		}
		for i, m := range head.Markers[:len(head.Markers)-1] {
			if span := head.Markers[i+1] - m; span != 4 {
				t.Fatalf("option %d spans %d ids, want 4", i, span)
			}
		}
		if !head.Stats.Collapsed() || head.Stats.OptionsDistinct != 1 {
			t.Errorf("options sharing a 3-token prefix must collapse: %+v", head.Stats)
		}
		// the instruction keeps at least 8 ids however the options crowd it
		if first := head.Markers[0]; first < 1+minInstructionTokens+1 {
			t.Errorf("the instruction kept %d ids, want at least %d", first-2, minInstructionTokens)
		}
	})
	t.Run("cap is floor((head-16)/k)", func(t *testing.T) {
		labels := make([]string, 5)
		for i := range labels {
			labels[i] = strings.Repeat(string(rune('a'+i)), 60)
		}
		q := schema.MustQuestion(schema.ChoiceOf("p", labels...))
		head, err := sequence.BuildHead(tok, q, 16+5*13+4) // (head-16)/5 = 13.8 -> 13
		if err != nil {
			t.Fatal(err)
		}
		if head.Stats.TokensPerOption == nil || *head.Stats.TokensPerOption != 13 {
			t.Errorf("TokensPerOption = %v, want 13", head.Stats.TokensPerOption)
		}
	})
}

func TestBuildHeadRefuses(t *testing.T) {
	tok := toy(t)
	if _, err := sequence.BuildHead(tok, schema.Question{}, 192); err == nil {
		t.Error("the zero Question was accepted")
	}
	if _, err := sequence.BuildHead(nil, schema.MustQuestion(schema.Noul("x")), 192); err == nil {
		t.Error("a nil tokenizer was accepted")
	}
}

func TestBuildTruncation(t *testing.T) {
	tok := toy(t)
	q := schema.MustQuestion(schema.Noul("x"))
	head, err := sequence.BuildHead(tok, q, 192)
	if err != nil {
		t.Fatal(err)
	}
	state := make([]int32, 100)
	for i := range state {
		state[i] = int32(i)
	}
	maxLen := len(head.IDs) + 1 + 30 // room for 30 state tokens
	b := sequence.Budget{MaxLen: maxLen, HeadMaxLen: 192}

	cases := []struct {
		name string
		keep sequence.Keep
		want []int32
	}{
		{"start kept", sequence.KeepStart, state[:30]},
		{"zero Keep is start", "", state[:30]},
		{"end kept", sequence.KeepEnd, state[70:]},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			seq, err := sequence.Build(tok, q, sequence.State{IDs: state, Keep: c.keep}, b)
			if err != nil {
				t.Fatal(err)
			}
			wantIDs := append(append(slices.Clone(head.IDs), c.want...), fixtures.ToySEP)
			if !slices.Equal(seq.IDs, wantIDs) {
				t.Errorf("ids = %v\nwant  %v", seq.IDs, wantIDs)
			}
			wantTr := sequence.Truncation{StateTokens: 100, StateTokensUsed: 30, StateTokensDropped: 70, Truncated: true}
			if seq.Truncation != wantTr {
				t.Errorf("Truncation = %+v, want %+v", seq.Truncation, wantTr)
			}
			if len(seq.IDs) > b.MaxLen {
				t.Errorf("%d ids exceed MaxLen %d", len(seq.IDs), b.MaxLen)
			}
		})
	}

	t.Run("no room at all keeps none of the state, from either end", func(t *testing.T) {
		for _, keep := range []sequence.Keep{sequence.KeepStart, sequence.KeepEnd} {
			seq, err := sequence.Build(tok, q, sequence.State{IDs: state, Keep: keep}, sequence.Budget{MaxLen: len(head.IDs), HeadMaxLen: 192})
			if err != nil {
				t.Fatal(err)
			}
			if seq.Truncation.StateTokensUsed != 0 || len(seq.IDs) != len(head.IDs) {
				t.Errorf("keep %q: used %d state tokens, %d ids", keep, seq.Truncation.StateTokensUsed, len(seq.IDs))
			}
		}
	})
	t.Run("a state that fits is not truncated", func(t *testing.T) {
		seq, err := sequence.Build(tok, q, sequence.State{IDs: state[:5]}, sequence.Budget{MaxLen: 512, HeadMaxLen: 192})
		if err != nil {
			t.Fatal(err)
		}
		if seq.Truncation.Truncated || seq.Truncation.StateTokensDropped != 0 || len(seq.IDs) != len(head.IDs)+5+1 {
			t.Errorf("seq = %+v", seq.Truncation)
		}
	})
	t.Run("markers past MaxLen are refused, as the agent refuses them", func(t *testing.T) {
		labels := make([]string, 10)
		for i := range labels {
			labels[i] = strings.Repeat("o", 30) + string(rune('a'+i))
		}
		many := schema.MustQuestion(schema.ChoiceOf("p", labels...))
		_, err := sequence.Build(tok, many, sequence.State{}, sequence.Budget{MaxLen: 100, HeadMaxLen: 192})
		var me *sequence.MarkersError
		if !errors.As(err, &me) || me.Total != 10 || me.Kept >= 10 || me.MaxLen != 100 || me.HeadMaxLen != 192 {
			t.Fatalf("err = %v", err)
		}
		want := fmt.Sprintf("only %d of its 10 option markers fit in max_len=100 with head_max_len=192 spent on the question; "+
			"lower head_max_len, raise max_len, or use fewer options", me.Kept)
		if err.Error() != want {
			t.Errorf("text\n got %q\nwant %q", err.Error(), want)
		}
		me.ID = "it's"
		if got := me.Error(); !strings.HasPrefix(got, `question "it's": only `) {
			t.Errorf("with an id: %q", got)
		}
	})
}

func TestBuildRefuses(t *testing.T) {
	tok := toy(t)
	q := schema.MustQuestion(schema.Noul("x"))
	for _, b := range []sequence.Budget{{}, {MaxLen: 512}, {HeadMaxLen: 192}, {MaxLen: -1, HeadMaxLen: 192}} {
		if _, err := sequence.Build(tok, q, sequence.State{}, b); err == nil {
			t.Errorf("Build accepted %+v", b)
		}
		if _, err := sequence.StateRoom(tok, q, b); err == nil {
			t.Errorf("StateRoom accepted %+v", b)
		}
		if _, err := sequence.PlanWindows(tok, []schema.Question{q}, b, sequence.WindowRequest{}); err == nil {
			t.Errorf("PlanWindows accepted %+v", b)
		}
	}
}

func TestStateRoomMatchesBuild(t *testing.T) {
	tok := toy(t)
	q := schema.MustQuestion(schema.ChoiceOf("which one", "alpha", "beta", "gamma"))
	b := sequence.Budget{MaxLen: 200, HeadMaxLen: 100}
	room, err := sequence.StateRoom(tok, q, b)
	if err != nil {
		t.Fatal(err)
	}
	seq, err := sequence.Build(tok, q, sequence.State{IDs: make([]int32, 1000)}, b)
	if err != nil {
		t.Fatal(err)
	}
	if seq.Truncation.StateTokensUsed != room || len(seq.IDs) != b.MaxLen {
		t.Errorf("StateRoom = %d but Build kept %d and filled %d of %d", room, seq.Truncation.StateTokensUsed, len(seq.IDs), b.MaxLen)
	}
}

func TestDefaultWindow(t *testing.T) {
	cases := []struct {
		b    sequence.Budget
		want int
	}{
		{sequence.Budget{MaxLen: 512, HeadMaxLen: 192}, 312},
		{sequence.Budget{MaxLen: 1024, HeadMaxLen: 256}, 760},
		{sequence.Budget{MaxLen: 100, HeadMaxLen: 90}, 64}, // the floor binds
	}
	for _, c := range cases {
		if got := c.b.DefaultWindow(); got != c.want {
			t.Errorf("%+v.DefaultWindow() = %d, want %d", c.b, got, c.want)
		}
	}
}

func TestPlanWindowsBranches(t *testing.T) {
	tok := toy(t)
	b := sequence.Budget{MaxLen: 512, HeadMaxLen: 192}
	q := schema.MustQuestion(schema.Noul("x"))
	t.Run("no questions: the window stands", func(t *testing.T) {
		plan, err := sequence.PlanWindows(tok, nil, b, sequence.WindowRequest{Window: 90})
		if err != nil || plan.Window != 90 || plan.Stride != 45 || plan.Room != 90 || plan.Clamped() {
			t.Errorf("plan = %+v, %v", plan, err)
		}
	})
	t.Run("defaults", func(t *testing.T) {
		plan, err := sequence.PlanWindows(tok, []schema.Question{q}, b, sequence.WindowRequest{})
		if err != nil || plan.Window != 312 || plan.Stride != 156 || plan.Clamped() {
			t.Errorf("plan = %+v, %v", plan, err)
		}
	})
	t.Run("nothing to read is ErrNoRoom", func(t *testing.T) {
		_, err := sequence.PlanWindows(tok, []schema.Question{q}, sequence.Budget{MaxLen: 8, HeadMaxLen: 192}, sequence.WindowRequest{})
		if !errors.Is(err, sequence.ErrNoRoom) {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("a stride past the window is ErrStrideGap", func(t *testing.T) {
		_, err := sequence.PlanWindows(tok, []schema.Question{q}, b, sequence.WindowRequest{Window: 100, Stride: 101})
		if !errors.Is(err, sequence.ErrStrideGap) || !strings.Contains(err.Error(), "pass stride <= 100") {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("the smallest room decides", func(t *testing.T) {
		labels := make([]string, 60)
		for i := range labels {
			labels[i] = strings.Repeat("o", 20) + string(rune('a'+i%26)) + string(rune('a'+i/26))
		}
		tight := schema.MustQuestion(schema.ChoiceOf("p", labels...))
		one, err := sequence.PlanWindows(tok, []schema.Question{tight}, b, sequence.WindowRequest{})
		if err != nil {
			t.Fatal(err)
		}
		both, err := sequence.PlanWindows(tok, []schema.Question{q, tight}, b, sequence.WindowRequest{})
		if err != nil || both.Room != one.Room || both.Window != one.Window {
			t.Errorf("plan with a roomy question = %+v, alone %+v, %v", both, one, err)
		}
	})
}

func TestBatchCap(t *testing.T) {
	cases := []struct {
		name                          string
		windows, window, config, size int
		want                          int
	}{
		{"explicit batch size wins", 100, 10, 312, 8, 8},
		{"explicit batch size wins when unclamped", 100, 312, 312, 3, 3},
		{"not clamped: one pass", 100, 312, 312, 0, 0},
		{"wider than config: one pass", 100, 400, 312, 0, 0},
		{"mild clamp: one pass", 11, 43, 64, 0, 0},
		{"exactly 2x is still one pass", 2, 78, 312, 0, 0},
		{"just past 2x is capped", 3, 78, 312, 0, 1},
		{"half the window, many windows", 20, 78, 312, 0, 5},
		{"hard clamp is capped at the unclamped count", 413, 23, 312, 0, 30},
		{"negative batch size is unset", 413, 23, 312, -4, 30},
		{"a floor of one", 1, 1, 312, 0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := sequence.BatchCap(c.windows, c.window, c.config, c.size)
			if got != c.want || ok != (c.want > 0) {
				t.Errorf("BatchCap(%d, %d, %d, %d) = %d, %v; want %d", c.windows, c.window, c.config, c.size, got, ok, c.want)
			}
		})
	}
}

func TestBatchCapDoesNotOverflow(t *testing.T) {
	// 2^62 windows of 23 tokens against a 312-token default: n*11 overflows int64, and Python's
	// integers do not, so the answer is the exact ceiling, not a wrapped 1 or a None.
	n := 1 << 62
	got, ok := sequence.BatchCap(n, 23, 312, 0)
	want := new(big.Int).Mul(big.NewInt(int64(n)), big.NewInt(11))
	want.Add(want, big.NewInt(155)).Div(want, big.NewInt(156)) // ceil(n*11/156)
	if !ok || big.NewInt(int64(got)).Cmp(want) != 0 {
		t.Errorf("BatchCap(2^62, 23, 312) = %d, %v; want %v", got, ok, want)
	}
	// a mild clamp at that scale is still Python's None, not a wrapped number
	if got, ok := sequence.BatchCap(n, 300, 312, 0); ok {
		t.Errorf("BatchCap(2^62, 300, 312) = %d, want none", got)
	}
	if got, ok := sequence.BatchCap(math.MaxInt, 1, 1<<40, 0); !ok || got <= 0 {
		t.Errorf("BatchCap(MaxInt, 1, 2^40) = %d, %v", got, ok)
	}
}

func TestConcurrentUse(t *testing.T) {
	tok := toy(t)
	q := schema.MustQuestion(schema.ChoiceOf("which", "alpha", "beta"))
	st, err := sequence.EncodeState(tok, map[string]any{"k": "the in the"})
	if err != nil {
		t.Fatal(err)
	}
	want, err := sequence.Build(tok, q, st, sequence.Budget{MaxLen: 128, HeadMaxLen: 64})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				s, _ := sequence.EncodeState(tok, map[string]any{"k": "the in the"})
				got, err := sequence.Build(tok, q, s, sequence.Budget{MaxLen: 128, HeadMaxLen: 64})
				if err != nil || !slices.Equal(got.IDs, want.IDs) || !slices.Equal(got.Markers, want.Markers) {
					t.Error("concurrent Build disagreed")
					return
				}
			}
		}()
	}
	wg.Wait()
}
