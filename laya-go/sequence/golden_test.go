package sequence_test

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/internal/goldens"
	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/sequence"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

var checkpoints = []string{"english", "multilingual"}

var (
	loadMu sync.Mutex
	loaded = map[string]*tokenizer.Tokenizer{}
)

// checkpoint loads a shipped checkpoint's tokenizer once per test binary. Without
// LAYA_CHECKPOINTS the test is skipped, and CI fails the run on any skip, so a missing
// checkpoint cannot pass as green.
func checkpoint(tb testing.TB, name string) *tokenizer.Tokenizer {
	tb.Helper()
	root := fixtures.Checkpoints()
	if root == "" {
		tb.Skip("set LAYA_CHECKPOINTS to a directory holding the english and multilingual checkpoints")
	}
	loadMu.Lock()
	defer loadMu.Unlock()
	if tok, ok := loaded[name]; ok {
		return tok
	}
	tok, err := tokenizer.LoadFS(os.DirFS(root), name)
	if err != nil {
		tb.Fatalf("load the %s tokenizer: %v", name, err)
	}
	loaded[name] = tok
	return tok
}

func firstDifference[T comparable](got, want []T) string {
	for i := 0; i < min(len(got), len(want)); i++ {
		if got[i] != want[i] {
			return fmt.Sprintf("first difference at %d: got %v, want %v (lengths %d and %d)", i, got[i], want[i], len(got), len(want))
		}
	}
	return fmt.Sprintf("one is a prefix of the other (lengths %d and %d)", len(got), len(want))
}

func intsOf(tb testing.TB, v any) []int {
	tb.Helper()
	ids := fixtures.Ints(tb, v)
	out := make([]int, len(ids))
	for i, id := range ids {
		out[i] = int(id)
	}
	return out
}

func optionalOrder(tb testing.TB, v any) []int32 {
	tb.Helper()
	if v == nil {
		return nil
	}
	return fixtures.Ints(tb, v)
}

// TestSequencesGolden holds Build to build_sequence on 26 inputs per checkpoint chosen for the
// head budget's branches and every rendering rule: ids, markers, the option-collision count and
// the truncation report. The state is tokenized by EncodeState, as the agent does.
func TestSequencesGolden(t *testing.T) {
	golden := fixtures.Load(t, "sequences.json")
	for _, name := range checkpoints {
		tok := checkpoint(t, name)
		entry := fixtures.Obj(t, fixtures.Field(t, golden, name))
		cases := fixtures.Obj(t, entry.Must("cases"))
		if len(cases) != 26 {
			t.Fatalf("%s: the golden holds %d cases", name, len(cases))
		}
		for _, c := range cases {
			rec := fixtures.Obj(t, c.Value)
			t.Run(name+"/"+c.Key, func(t *testing.T) {
				q := goldens.Question(t, fixtures.Obj(t, rec.Must("question")), optionalOrder(t, rec.Must("option_order")))
				state, err := sequence.EncodeState(tok, goldens.Value(rec.Must("state")))
				if err != nil {
					t.Fatal(err)
				}
				if fixtures.Bool(t, rec.Must("truncate_left")) {
					state.Keep = sequence.KeepEnd
				}
				b := sequence.Budget{MaxLen: fixtures.Int(t, rec.Must("max_len")), HeadMaxLen: fixtures.Int(t, rec.Must("head_max_len"))}
				seq, err := sequence.Build(tok, q, state, b)
				wantMarkers := intsOf(t, rec.Must("markers"))
				optionCount := fixtures.Int(t, fixtures.Obj(t, rec.Must("stats")).Must("options"))
				var dropped *sequence.MarkersError
				if errors.As(err, &dropped) {
					// build_sequence returns the markers that fit and the agent refuses; Build does
					// both, so the refusal must name exactly the markers Python kept.
					if dropped.Kept != len(wantMarkers) || dropped.Total != optionCount || len(wantMarkers) >= optionCount {
						t.Errorf("MarkersError = %+v; Python kept %d of %d", *dropped, len(wantMarkers), optionCount)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if len(wantMarkers) != optionCount {
					t.Fatalf("Python dropped markers (%d of %d) and Build did not refuse", len(wantMarkers), optionCount)
				}
				if want := fixtures.Ints(t, rec.Must("ids")); !slices.Equal(seq.IDs, want) {
					t.Errorf("ids: %s", firstDifference(seq.IDs, want))
				}
				if !slices.Equal(seq.Markers, wantMarkers) {
					t.Errorf("markers = %v, want %v", seq.Markers, wantMarkers)
				}
				stats := fixtures.Obj(t, rec.Must("stats"))
				if got, want := seq.Stats.Options, fixtures.Int(t, stats.Must("options")); got != want {
					t.Errorf("Stats.Options = %d, want %d", got, want)
				}
				if got, want := seq.Stats.OptionsDistinct, fixtures.Int(t, stats.Must("options_distinct")); got != want {
					t.Errorf("Stats.OptionsDistinct = %d, want %d", got, want)
				}
				switch want := stats.Must("tokens_per_option"); {
				case want == nil && seq.Stats.TokensPerOption != nil:
					t.Errorf("TokensPerOption = %d, want none", *seq.Stats.TokensPerOption)
				case want != nil && (seq.Stats.TokensPerOption == nil || *seq.Stats.TokensPerOption != fixtures.Int(t, want)):
					t.Errorf("TokensPerOption = %v, want %v", seq.Stats.TokensPerOption, want)
				}
				tr := fixtures.Obj(t, rec.Must("truncation"))
				want := sequence.Truncation{
					StateTokens:        fixtures.Int(t, tr.Must("state_tokens")),
					StateTokensUsed:    fixtures.Int(t, tr.Must("state_tokens_used")),
					StateTokensDropped: fixtures.Int(t, tr.Must("state_tokens_dropped")),
					Truncated:          fixtures.Bool(t, tr.Must("truncated")),
				}
				if seq.Truncation != want {
					t.Errorf("Truncation = %+v, want %+v", seq.Truncation, want)
				}
				if seq.Stats.Collapsed() != (want.StateTokens < 0 || seq.Stats.OptionsDistinct < seq.Stats.Options) {
					t.Errorf("Collapsed() = %v with %d of %d options distinct", seq.Stats.Collapsed(), seq.Stats.OptionsDistinct, seq.Stats.Options)
				}
			})
		}
	}
}

// TestSpecialIDsGolden checks the ids Build places against the ones Python's tokenizer reports,
// because multilingual reuses <bos> and <eos> as CLS and SEP and nothing in the vocabulary says so.
func TestSpecialIDsGolden(t *testing.T) {
	golden := fixtures.Load(t, "sequences.json")
	for _, name := range checkpoints {
		tok := checkpoint(t, name)
		ids := fixtures.Obj(t, fixtures.Obj(t, fixtures.Field(t, golden, name)).Must("special_ids"))
		cls, _ := tok.CLS()
		sep, _ := tok.SEP()
		mask, _ := tok.Mask()
		pad, _ := tok.Pad()
		for role, got := range map[string]int32{"cls": cls, "sep": sep, "mask": mask, "pad": pad} {
			if want := int32(fixtures.Int(t, ids.Must(role))); got != want {
				t.Errorf("%s: %s id = %d, Python has %d", name, role, got, want)
			}
		}
		if got, want := tok.MaskToken(), fixtures.Str(t, ids.Must("mask_token")); got != want {
			t.Errorf("%s: mask token = %q, Python has %q", name, got, want)
		}
	}
}

// windowQuestion is the generator's _window_question: options of four words each.
func windowQuestion(tb testing.TB, options int) schema.Question {
	tb.Helper()
	opts := make([]schema.Option, options)
	for i := range opts {
		text := ""
		for w := 0; w < 4; w++ {
			if w > 0 {
				text += " "
			}
			text += fmt.Sprintf("criterion%d word%d", i, w)
		}
		opts[i] = schema.Option{Label: fmt.Sprintf("opt%03d", i), Criterion: text}
	}
	return schema.MustQuestion(schema.Choice("Pick one.", opts...))
}

// TestWindowPlanGolden holds StateRoom and PlanWindows to state_room and window_budget, warnings
// and refusals included as text, and BatchCap to window_batch_cap over the whole grid.
func TestWindowPlanGolden(t *testing.T) {
	golden := fixtures.Load(t, "window_plan.json")
	by := fixtures.Obj(t, golden.Must("by"))
	for _, name := range checkpoints {
		tok := checkpoint(t, name)
		entry := fixtures.Obj(t, fixtures.Field(t, by, name))
		maxLen, headMax := fixtures.Int(t, entry.Must("default_max_len")), fixtures.Int(t, entry.Must("default_head_max_len"))

		for _, r := range fixtures.List(t, entry.Must("state_room")) {
			rec := fixtures.Obj(t, r)
			options := fixtures.Int(t, rec.Must("options"))
			t.Run(fmt.Sprintf("%s/state_room/%d", name, options), func(t *testing.T) {
				q := windowQuestion(t, options)
				for key, b := range map[string]sequence.Budget{
					"room":       {MaxLen: maxLen, HeadMaxLen: headMax},
					"room_tight": {MaxLen: 256, HeadMaxLen: 128},
				} {
					got, err := sequence.StateRoom(tok, q, b)
					if err != nil {
						t.Fatal(err)
					}
					if want := fixtures.Int(t, rec.Must(key)); got != want {
						t.Errorf("%s = %d, want %d", key, got, want)
					}
				}
			})
		}

		for _, c := range fixtures.List(t, entry.Must("cases")) {
			rec := fixtures.Obj(t, c)
			t.Run(name+"/"+fixtures.Str(t, rec.Must("case")), func(t *testing.T) {
				var questions []schema.Question
				for _, n := range fixtures.Ints(t, rec.Must("options")) {
					questions = append(questions, windowQuestion(t, int(n)))
				}
				b := sequence.Budget{MaxLen: fixtures.Int(t, rec.Must("max_len")), HeadMaxLen: fixtures.Int(t, rec.Must("head_max_len"))}
				var req sequence.WindowRequest
				if w := rec.Must("window"); w != nil {
					req.Window = fixtures.Int(t, w)
				}
				if s := rec.Must("stride"); s != nil {
					req.Stride = fixtures.Int(t, s)
				}
				plan, err := sequence.PlanWindows(tok, questions, b, req)
				if want, refused := rec.Get("error"); refused {
					text := fixtures.Str(t, want)
					if err == nil {
						t.Fatalf("no error, Python refuses with %q", text)
					}
					if err.Error() != text {
						t.Errorf("error text\n got %q\nwant %q", err.Error(), text)
					}
					sentinel := sequence.ErrStrideGap
					if strings.Contains(text, "fill the whole sequence") {
						sentinel = sequence.ErrNoRoom
					}
					if !errors.Is(err, sentinel) {
						t.Errorf("error %v does not wrap %v", err, sentinel)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				res := fixtures.Obj(t, rec.Must("result"))
				got := [3]int{plan.Window, plan.Stride, plan.Room}
				want := [3]int{fixtures.Int(t, res.Must("window")), fixtures.Int(t, res.Must("stride")), fixtures.Int(t, res.Must("room"))}
				if got != want {
					t.Errorf("window, stride, room = %v, want %v", got, want)
				}
				wantWarnings := fixtures.Strings(t, rec.Must("warnings"))
				if !slices.Equal(plan.Warnings, wantWarnings) {
					t.Errorf("warnings\n got %q\nwant %q", plan.Warnings, wantWarnings)
				}
				if plan.Clamped() != (len(wantWarnings) > 0) {
					t.Errorf("Clamped() = %v with %d warnings", plan.Clamped(), len(wantWarnings))
				}
			})
		}
	}

	grid := fixtures.List(t, golden.Must("batch_cap"))
	if len(grid) < 800 {
		t.Fatalf("the batch_cap grid holds %d points", len(grid))
	}
	capped := 0
	for _, g := range grid {
		rec := fixtures.Obj(t, g)
		in := [4]int{fixtures.Int(t, rec.Must("n_windows")), fixtures.Int(t, rec.Must("window")), fixtures.Int(t, rec.Must("config_budget"))}
		if b := rec.Must("batch_size"); b != nil {
			in[3] = fixtures.Int(t, b)
		}
		want, wantOK := 0, false // Python's None
		if c := rec.Must("cap"); c != nil {
			want, wantOK = fixtures.Int(t, c), true
		}
		if got, ok := sequence.BatchCap(in[0], in[1], in[2], in[3]); got != want || ok != wantOK {
			t.Errorf("BatchCap%v = %d, %v; Python has %d, %v", in, got, ok, want, wantOK)
		}
		if want > 0 && in[3] <= 0 {
			capped++
		}
	}
	if capped == 0 {
		t.Error("no grid point reached the capping branch")
	}
}
