package shortlist

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/internal/fixtures"
)

type counting struct {
	calls [][]string
	next  func(texts []string) [][]float64
}

func (c *counting) embed(ctx context.Context, texts []string) ([][]float64, error) {
	c.calls = append(c.calls, append([]string(nil), texts...))
	if c.next != nil {
		return c.next(texts), nil
	}
	return stubEmbed(ctx, texts)
}

func TestCacheCountersMatchPython(t *testing.T) {
	steps := fixtures.List(t, fixtures.Load(t, "shortlist.json").Must("cache_counters"))
	inner := &counting{}
	c, err := Cached(inner.embed, 64)
	if err != nil {
		t.Fatal(err)
	}
	for _, raw := range steps {
		s := fixtures.Obj(t, raw)
		if _, err := c.Embed(context.Background(), fixtures.Strings(t, s.Must("texts"))); err != nil {
			t.Fatal(err)
		}
		want := CacheInfo{Size: fixtures.Int(t, s.Must("size")), MaxSize: fixtures.Int(t, s.Must("maxsize")),
			Hits: fixtures.Int(t, s.Must("hits")), Misses: fixtures.Int(t, s.Must("misses"))}
		if got := c.Info(); got != want {
			t.Errorf("after %v: %+v, want %+v", s.Must("texts"), got, want)
		}
	}
	if len(steps) != 3 {
		t.Errorf("%d steps", len(steps))
	}
	// Each cold text reached the embedder once, in first-seen order, in one call per step.
	want := [][]string{{"a", "b"}, {"c"}, {"q"}}
	if !reflect.DeepEqual(inner.calls, want) {
		t.Errorf("embedder calls %q, want %q", inner.calls, want)
	}
}

func TestCacheRowsAndEviction(t *testing.T) {
	inner := &counting{}
	c, _ := Cached(inner.embed, 2)
	ctx := context.Background()
	got, err := c.Embed(ctx, []string{"a", "b", "a"})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got[0], got[2]) || reflect.DeepEqual(got[0], got[1]) {
		t.Errorf("rows %v", got)
	}
	want, _ := stubEmbed(ctx, []string{"a"})
	for i := range want[0] {
		if d := got[0][i] - want[0][i]; d > 1e-6 || d < -1e-6 {
			t.Errorf("component %d: %v vs %v: more than float32 apart", i, got[0][i], want[0][i])
		}
	}
	got[0][0] = 99 // a caller cannot reach into the cache
	again, _ := c.Embed(ctx, []string{"a"})
	if again[0][0] == 99 {
		t.Error("the cache returned its own row")
	}
	// "a" was just used, so "b" is the oldest and "c" evicts it.
	c.Embed(ctx, []string{"c"})
	n := len(inner.calls)
	c.Embed(ctx, []string{"a"})
	if len(inner.calls) != n {
		t.Error("a was used last but was evicted")
	}
	c.Embed(ctx, []string{"b"})
	if len(inner.calls) != n+1 {
		t.Error("b was not evicted")
	}
	if info := c.Info(); info.Size != 2 {
		t.Errorf("size %d, want 2", info.Size)
	}
	// More new texts than the bound: all are returned, the newest are kept.
	rows, err := c.Embed(ctx, []string{"x", "y", "z"})
	if err != nil || len(rows) != 3 || c.Info().Size != 2 {
		t.Errorf("rows %d, err %v, size %d", len(rows), err, c.Info().Size)
	}
	if rows, err := c.Embed(ctx, nil); err != nil || len(rows) != 0 {
		t.Errorf("no texts: %v, %v", rows, err)
	}
	c.Clear()
	if info := c.Info(); info != (CacheInfo{MaxSize: 2}) {
		t.Errorf("after Clear: %+v", info)
	}
}

func TestCacheNonFiniteIsCachedAsZero(t *testing.T) {
	inner := &counting{next: func(texts []string) [][]float64 {
		rows := make([][]float64, len(texts))
		for i := range rows {
			rows[i] = []float64{inf(), 1}
		}
		return rows
	}}
	c, _ := Cached(inner.embed, 8)
	for i := 0; i < 2; i++ {
		rows, err := c.Embed(context.Background(), []string{"a"})
		if err != nil || rows[0][0] != 0 || rows[0][1] != 1 {
			t.Fatalf("%v, %v", rows, err)
		}
	}
	if len(inner.calls) != 1 {
		t.Errorf("the repaired row was not cached: %d calls", len(inner.calls))
	}
}

func inf() float64 { var z float64; return 1 / z }

func TestCacheRefusals(t *testing.T) {
	ctx := context.Background()
	if _, err := Cached(nil, 4); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil embedder: %v", err)
	}
	for _, size := range []int{0, -1} {
		if _, err := Cached(stubEmbed, size); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("size %d: %v", size, err)
		}
	}
	width := 2
	inner := &counting{next: func(texts []string) [][]float64 {
		rows := make([][]float64, len(texts))
		for i := range rows {
			rows[i] = make([]float64, width)
			rows[i][0] = 1
		}
		return rows
	}}
	c, _ := Cached(inner.embed, 8)
	if _, err := c.Embed(ctx, []string{"a"}); err != nil {
		t.Fatal(err)
	}
	width = 3
	if _, err := c.Embed(ctx, []string{"b"}); !errors.Is(err, ErrEmbedding) {
		t.Errorf("dimension drift: %v", err)
	}
	if info := c.Info(); info.Size != 1 {
		t.Errorf("a refused call was cached: %+v", info)
	}
	c.Clear()
	if _, err := c.Embed(ctx, []string{"b"}); err != nil {
		t.Errorf("after Clear: %v", err)
	}
	bad, _ := Cached(func(context.Context, []string) ([][]float64, error) { return [][]float64{{1}}, nil }, 8)
	if _, err := bad.Embed(ctx, []string{"a", "b"}); !errors.Is(err, ErrEmbedding) || bad.Info().Size != 0 {
		t.Errorf("bad shape: %v, %+v", err, bad.Info())
	}
	boom := errors.New("boom")
	failing, _ := Cached(func(context.Context, []string) ([][]float64, error) { return nil, boom }, 8)
	if _, err := failing.Embed(ctx, []string{"a"}); !errors.Is(err, boom) || failing.Info().Size != 0 {
		t.Errorf("embedder error: %v", err)
	}
}

func TestCacheConcurrent(t *testing.T) {
	c, _ := Cached(stubEmbed, 16)
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				if _, err := c.Embed(context.Background(), []string{"a", "b", string(rune('c' + i%10))}); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
	if info := c.Info(); info.Hits+info.Misses != 8*50*3 || info.Size > 16 {
		t.Errorf("%+v", info)
	}
}

func TestCacheFloat32OverflowIsZero(t *testing.T) {
	inner := &counting{next: func(texts []string) [][]float64 {
		return [][]float64{{1e300, -1e300, 2}}
	}}
	c, _ := Cached(inner.embed, 4)
	for i := 0; i < 2; i++ {
		rows, err := c.Embed(context.Background(), []string{"a"})
		if err != nil || rows[0][0] != 0 || rows[0][1] != 0 || rows[0][2] != 2 {
			t.Fatalf("%v, %v", rows, err)
		}
	}
}
