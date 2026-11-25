package sequence

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"slices"

	"github.com/NandhaKishorM/laya/laya-go/schema"
	"github.com/NandhaKishorM/laya/laya-go/tokenizer"
)

// ErrNoRoom is wrapped when the questions' heads fill the whole sequence, so no window can
// carry any of the state.
var ErrNoRoom = errors.New("questions leave no room for the state")

// ErrStrideGap is wrapped when an explicit stride steps past the window, so the tokens between
// two windows would be read by none.
var ErrStrideGap = errors.New("stride steps past the window")

// WindowRequest is what the caller asked for. Zero and negative values mean "unset", as in
// laya, not "invalid".
type WindowRequest struct {
	// Window is the state tokens per window; unset means Budget.DefaultWindow.
	Window int
	// Stride is the step between windows; unset means half the effective window.
	Stride int
}

// WindowPlan is a sized scan.
type WindowPlan struct {
	// Window is the state tokens per window, after any clamp.
	Window int
	// Stride is the step between windows, at most Window, so every token is read.
	Stride int
	// Room is the smallest state room of the questions, or the requested window when there
	// were no questions to fit.
	Room int
	// Warnings are laya's RuntimeWarning messages for each clamp, verbatim and in order; empty
	// when nothing was clamped.
	Warnings []string
}

// Clamped reports whether the window or stride was reduced.
func (p WindowPlan) Clamped() bool { return len(p.Warnings) > 0 }

// PlanWindows is laya's window_budget: the window and stride for scanning a state longer than
// one sequence against questions.
//
// The window is capped at the smallest StateRoom of questions, because the windows are scored
// for every question in shared passes. An explicit window wider than that is clamped with a
// warning; the default window is clamped silently unless it shrinks by 2x or more. A default
// stride is half the effective window. An explicit stride past the effective window is reduced
// with a warning when it was valid for the window asked for, and refused with ErrStrideGap
// otherwise. Questions that leave no room are refused with ErrNoRoom. A refusal's text is
// laya's ValueError message verbatim.
func PlanWindows(tok *tokenizer.Tokenizer, questions []schema.Question, b Budget, req WindowRequest) (WindowPlan, error) {
	if err := checkBudget(b); err != nil {
		return WindowPlan{}, err
	}
	rooms := make([]int, len(questions))
	for i, q := range questions {
		room, err := StateRoom(tok, q, b)
		if err != nil {
			return WindowPlan{}, err
		}
		rooms[i] = room
	}
	requested := req.Window
	if requested <= 0 {
		requested = b.DefaultWindow()
	}
	size := requested
	room := size // nothing to fit: the window stands
	if len(rooms) > 0 {
		room = slices.Min(rooms)
	}
	if room <= 0 {
		return WindowPlan{}, &windowError{kind: ErrNoRoom, msg: fmt.Sprintf(
			"predict_long: the questions' options fill the whole sequence (max_len=%d,"+
				" head_max_len=%d), leaving no room for the state; no window can carry any of it."+
				" A label set this large is what laya.shortlist.predict_shortlist is for",
			b.MaxLen, b.HeadMaxLen)}
	}
	var warnings []string
	if size > room {
		switch {
		case req.Window > 0:
			warnings = append(warnings, fmt.Sprintf(
				"laya: predict_long: window=%d is wider than the %d state tokens these questions"+
					" leave inside max_len=%d, so every window would be truncated to %d on the way to"+
					" the model; scanning with window=%d instead", size, room, b.MaxLen, room, room))
		case size >= room*windowClampWarnRatio:
			warnings = append(warnings, fmt.Sprintf(
				"laya: predict_long: these questions leave only %d of max_len=%d for the state"+
					" (their heads take the rest), so the scan window is capped %d -> %d and roughly"+
					" %.1fx as many windows -- each a full forward pass -- are needed to read the"+
					" document. Fewer or shorter options, a larger max_len, or"+
					" laya.shortlist.predict_shortlist for a large label set will all cost less than"+
					" scanning at this width", room, b.MaxLen, size, room, float64(size)/float64(room)))
		}
		size = room
	}
	step := req.Stride
	if step <= 0 {
		step = max(1, size/2)
	}
	if step > size {
		if size < requested && req.Stride > 0 && req.Stride <= requested {
			// The window the caller asked for was reduced, and their stride was valid for it: the
			// library's clamp, not their mistake.
			warnings = append(warnings, fmt.Sprintf(
				"laya: predict_long: stride=%d was a 50%% step for the window=%d you asked for, but"+
					" the window was reduced to %d to fit the room these questions leave; scanning with"+
					" stride=%d instead", step, requested, size, max(1, size/2)))
			step = max(1, size/2)
		} else {
			why := ""
			if size < requested {
				why = " these questions leave room for"
			}
			return WindowPlan{}, &windowError{kind: ErrStrideGap, msg: fmt.Sprintf(
				"predict_long: stride=%d steps past the %d-token window%s, so %d tokens between"+
					" every pair of windows would be read by no window at all; pass stride <= %d",
				step, size, why, step-size, size)}
		}
	}
	return WindowPlan{Window: size, Stride: step, Room: room, Warnings: warnings}, nil
}

// windowClampWarnRatio is how far the default window may be cut before PlanWindows says so: a
// small clamp is ordinary, and one of 2x or more doubles the forward passes of the scan.
const windowClampWarnRatio = 2

// windowBatchBlowup is how much capping the window may multiply the window count before
// BatchCap starts chunking.
const windowBatchBlowup = 2

// windowError carries laya's ValueError text and wraps the sentinel a caller tests for.
type windowError struct {
	kind error
	msg  string
}

func (e *windowError) Error() string { return e.msg }
func (e *windowError) Unwrap() error { return e.kind }

// BatchCap is laya's window_batch_cap: how many windows one forward pass may carry. ok is false
// where Python returns None, "every window in one pass".
//
// An explicit batchSize > 0 is always returned. Otherwise there is no cap unless clamping the
// window to window (from configWindow, the Budget.DefaultWindow) multiplied the window count by
// more than 2x; then the pass is bounded at the count the unclamped window would have produced,
// ceil(nWindows*max(1, window/2) / max(1, configWindow/2)). The arithmetic does not overflow:
// Python's integers do not, and a product of two large int values is computed exactly.
func BatchCap(nWindows, window, configWindow, batchSize int) (cap int, ok bool) {
	if batchSize > 0 {
		return batchSize, true
	}
	if window >= configWindow {
		return 0, false // not capped: one pass, exactly as before
	}
	num := new(big.Int).Mul(big.NewInt(int64(nWindows)), big.NewInt(int64(max(1, floorDiv(window, 2)))))
	den := big.NewInt(int64(max(1, floorDiv(configWindow, 2))))
	// ceil(num/den) == -floor(-num/den); big.Int.Div is floor for a positive divisor.
	unclamped := new(big.Int).Neg(new(big.Int).Div(new(big.Int).Neg(num), den))
	if unclamped.Sign() < 1 {
		unclamped.SetInt64(1)
	}
	// nWindows <= 2*unclamped
	limit := new(big.Int).Mul(unclamped, big.NewInt(windowBatchBlowup))
	if big.NewInt(int64(nWindows)).Cmp(limit) <= 0 {
		return 0, false // a mild cap: keep the single shared pass
	}
	if !unclamped.IsInt64() || unclamped.Int64() > int64(math.MaxInt) {
		return math.MaxInt, true // unreachable: the cap is below nWindows
	}
	return int(unclamped.Int64()), true
}
