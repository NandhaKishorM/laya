package router_test

import (
	"os"
	"reflect"
	"strconv"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
	"github.com/NandhaKishorM/laya/laya-go/router"
)

// files reads a golden {path: digest} object; JSON null is a nil map, {} an empty one.
func files(t *testing.T, v any) map[string]string {
	t.Helper()
	if v == nil {
		return nil
	}
	out := map[string]string{}
	for _, m := range fixtures.Obj(t, v) {
		out[m.Key] = fixtures.Str(t, m.Value)
	}
	return out
}

func perModel(t *testing.T, v any) map[string]map[string]string {
	t.Helper()
	if v == nil {
		return nil
	}
	out := map[string]map[string]string{}
	for _, m := range fixtures.Obj(t, v) {
		out[m.Key] = files(t, m.Value)
	}
	return out
}

func setEnv(t *testing.T, v any) {
	t.Helper()
	t.Setenv("LAYA_SHA256_DIGESTS", "")
	if v == nil {
		os.Unsetenv("LAYA_SHA256_DIGESTS")
		return
	}
	os.Setenv("LAYA_SHA256_DIGESTS", fixtures.Str(t, v))
}

// replayDigestRow checks one router_digests row against the Router and says whether it was an
// error row.
func replayDigestRow(t *testing.T, row fixtures.Object) (isErr bool) {
	t.Helper()
	setEnv(t, row.Must("env"))
	var opts []router.Option
	if arg := perModel(t, row.Must("argument")); arg != nil {
		opts = append(opts, router.WithSHA256Digests(arg))
	}
	r, err := router.New(opts...)
	if want, bad := row.Get("error"); bad {
		if err == nil || err.Error() != fixtures.Str(t, want) {
			t.Fatalf("New error = %v, want %q", err, want)
		}
		return true
	}
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	table := map[router.Checkpoint]map[string]string{}
	for k, v := range perModel(t, row.Must("table")) {
		table[router.Checkpoint(k)] = v
	}
	got := r.SHA256Digests()
	if len(got) != len(table) {
		t.Fatalf("table = %v, want %v", got, table)
	}
	for k, want := range table {
		g, ok := got[k]
		if !ok || (g == nil) != (want == nil) || (want != nil && !reflect.DeepEqual(g, want)) {
			t.Fatalf("table[%s] = %#v, want %#v", k, g, want)
		}
	}
	shared := files(t, row.Must("shared"))
	for _, m := range fixtures.Obj(t, row.Must("expected")) {
		want := files(t, m.Value)
		got, err := r.ExpectedSHA256(m.Key, shared)
		if err != nil || (got == nil) != (want == nil) || (want != nil && !reflect.DeepEqual(got, want)) {
			t.Fatalf("ExpectedSHA256(%s) = %#v, %v, want %#v", m.Key, got, err, want)
		}
	}
	return false
}

func TestRouterDigestsGolden(t *testing.T) {
	root := fixtures.Load(t, "go_extra.json")
	rows := fixtures.List(t, root.Must("router_digests"))
	if len(rows) != 1620 {
		t.Fatalf("%d rows", len(rows))
	}
	errs := 0
	for i, rv := range rows {
		row := fixtures.Obj(t, rv)
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			if replayDigestRow(t, row) {
				errs++
			}
		})
	}
	if errs != 660 {
		t.Fatalf("%d error rows, want 660", errs)
	}
}
