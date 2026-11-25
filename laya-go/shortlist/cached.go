package shortlist

import (
	"container/list"
	"context"
	"fmt"
	"math"
	"sync"
)

// DefaultCacheSize is the string count Cached keeps when the caller does not choose.
const DefaultCacheSize = 4096

// CacheInfo is what a CachedEmbedder reports. Hits and Misses count occurrences, so their sum is
// the number of texts looked up.
type CacheInfo struct {
	Size, MaxSize, Hits, Misses int
}

type cacheEntry struct {
	key string
	row []float32
}

// CachedEmbedder caches an Embedder's output per input string under an LRU bound: cached_embed_fn.
// Use its Embed method as the Embedder. It is safe for concurrent use; the lock covers only the
// cache, never the embedding call.
//
// Texts missing from the cache are deduplicated and embedded in one call. Rows are stored as
// float32, so a wrapped first call can differ from the unwrapped one in its last digits. A
// non-finite value becomes 0 on the way in and the repaired row is cached; Clear the cache if the
// model behind the embedder changes. Nothing is cached when the embedder fails or returns a bad
// shape.
type CachedEmbedder struct {
	inner  Embedder
	max    int
	mu     sync.Mutex
	lru    *list.List // front: most recently used
	byKey  map[string]*list.Element
	hits   int
	misses int
}

// Cached wraps embed. maxSize bounds the strings held (DefaultCacheSize is the usual value); a
// size below 1 is refused, as in Python.
func Cached(embed Embedder, maxSize int) (*CachedEmbedder, error) {
	if embed == nil {
		return nil, invalid("embedder must not be nil")
	}
	if maxSize < 1 {
		return nil, invalid("maxsize must be a positive integer, got %d", maxSize)
	}
	return &CachedEmbedder{inner: embed, max: maxSize, lru: list.New(), byKey: map[string]*list.Element{}}, nil
}

// Embed is the Embedder: one row per text, from the cache where it can.
func (c *CachedEmbedder) Embed(ctx context.Context, texts []string) ([][]float64, error) {
	if len(texts) == 0 {
		return [][]float64{}, nil
	}
	found := make(map[string][]float32, len(texts))
	c.mu.Lock()
	hits := 0
	for _, key := range texts {
		if el, ok := c.byKey[key]; ok {
			c.lru.MoveToFront(el)
			found[key] = el.Value.(*cacheEntry).row
			hits++
		}
	}
	c.hits += hits
	c.misses += len(texts) - hits
	var missing []string
	queued := map[string]bool{}
	for _, key := range texts {
		if _, ok := found[key]; !ok && !queued[key] {
			queued[key] = true
			missing = append(missing, key)
		}
	}
	c.mu.Unlock()

	if len(missing) > 0 {
		fresh, err := embeddings(ctx, c.inner, missing)
		if err != nil {
			return nil, err
		}
		rows := make([][]float32, len(fresh))
		for i, r := range fresh {
			rows[i] = make([]float32, len(r))
			for j, v := range r {
				rows[i][j] = float32(v)
				if f := rows[i][j]; f > math.MaxFloat32 || f < -math.MaxFloat32 { // float32 overflow, then nan_to_num
					rows[i][j] = 0
				}
			}
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		if c.lru.Len() > 0 {
			held := len(c.lru.Front().Value.(*cacheEntry).row)
			if len(rows[0]) != held {
				return nil, fmt.Errorf("%w: embedder returned dim %d, but the cache holds dim %d; call Clear if the model behind it changed",
					ErrEmbedding, len(rows[0]), held)
			}
		}
		for i, key := range missing {
			if el, ok := c.byKey[key]; ok { // another caller stored it meanwhile
				el.Value.(*cacheEntry).row = rows[i]
				c.lru.MoveToFront(el)
			} else {
				c.byKey[key] = c.lru.PushFront(&cacheEntry{key: key, row: rows[i]})
			}
			for c.lru.Len() > c.max {
				oldest := c.lru.Back()
				c.lru.Remove(oldest)
				delete(c.byKey, oldest.Value.(*cacheEntry).key)
			}
			found[key] = rows[i]
		}
	}
	out := make([][]float64, len(texts))
	for i, key := range texts {
		row := found[key]
		out[i] = make([]float64, len(row))
		for j, v := range row {
			out[i][j] = float64(v)
		}
	}
	return out, nil
}

// Info reports the cache counters.
func (c *CachedEmbedder) Info() CacheInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return CacheInfo{Size: c.lru.Len(), MaxSize: c.max, Hits: c.hits, Misses: c.misses}
}

// Clear empties the cache and zeroes the counters.
func (c *CachedEmbedder) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lru.Init()
	c.byKey = map[string]*list.Element{}
	c.hits, c.misses = 0, 0
}
