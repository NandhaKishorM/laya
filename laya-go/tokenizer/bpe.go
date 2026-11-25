package tokenizer

import (
	"container/heap"
	"fmt"
	"unicode/utf8"
)

// heapMinSymbols is the piece length from which merging uses a priority queue; shorter pieces
// are rescanned, which is faster at that size. The two give the same result.
const heapMinSymbols = 32

type mergeRule struct {
	rank int32
	id   int32 // the id of the merged symbol
}

// symbol is a stretch of a piece with its vocabulary id, or -1 for a character the vocabulary
// lacks.
type symbol struct {
	start, end int32
	id         int32
}

// smallRunes is how many leading code points have a direct id table. Byte-level pieces are made
// of the 256 stand-in characters (all below U+0144), so nearly every initial symbol skips the
// string hash.
const smallRunes = 0x180

type bpe struct {
	small        [smallRunes]int32 // id of the single-character token for r, or -1
	vocab        map[string]int32
	merges       map[uint64]mergeRule
	unkID        int32
	hasUnk       bool
	fuseUnk      bool
	ignoreMerges bool
	byteIDs      *[256]int32 // byte fallback ids, nil when the checkpoint has none
}

func pairKey(left, right int32) uint64 {
	return uint64(uint32(left))<<32 | uint64(uint32(right))
}

// newBPE indexes the merge table. Each merge's operands and result must be in the vocabulary,
// which is what lets merging work on ids instead of strings.
func newBPE(vocab map[string]int32, merges [][2]string, unk *string, fuseUnk, ignoreMerges, byteFallback bool) (*bpe, error) {
	b := &bpe{vocab: vocab, fuseUnk: fuseUnk, ignoreMerges: ignoreMerges,
		merges: make(map[uint64]mergeRule, len(merges))}
	for r := range b.small {
		id, ok := vocab[string(rune(r))]
		if !ok {
			id = -1
		}
		b.small[r] = id
	}
	if unk != nil {
		id, ok := vocab[*unk]
		if !ok {
			return nil, fmt.Errorf("the tokenizer names %s as its unknown token, but neither the "+
				"vocabulary nor the added-token table has an entry for it", *unk)
		}
		b.unkID, b.hasUnk = id, true
	}
	for rank, pair := range merges {
		left, okL := vocab[pair[0]]
		right, okR := vocab[pair[1]]
		if !okL || !okR {
			missing := pair[0]
			if okL {
				missing = pair[1]
			}
			return nil, fmt.Errorf("merge at rank %d uses the symbol %q, which the vocabulary does "+
				"not contain; merging on ids would be unsound", rank, missing)
		}
		merged, ok := vocab[pair[0]+pair[1]]
		if !ok {
			return nil, fmt.Errorf("merge at rank %d produces %q, which the vocabulary does not contain",
				rank, pair[0]+pair[1])
		}
		key := pairKey(left, right)
		if _, dup := b.merges[key]; !dup { // the lowest rank wins
			b.merges[key] = mergeRule{int32(rank), merged}
		}
	}
	if byteFallback {
		var ids [256]int32
		for v := range ids {
			id, ok := vocab[fmt.Sprintf("<0x%02X>", v)]
			if !ok {
				id = -1
			}
			ids[v] = id
		}
		b.byteIDs = &ids
	}
	return b, nil
}

func (b *bpe) rule(left, right symbol) (mergeRule, bool) {
	if left.id < 0 || right.id < 0 {
		return mergeRule{}, false
	}
	r, ok := b.merges[pairKey(left.id, right.id)]
	return r, ok
}

// encodePiece appends the ids of one pre-tokenized piece to out.
func (b *bpe) encodePiece(piece string, out []int32) []int32 {
	if piece == "" {
		return out
	}
	if b.ignoreMerges {
		if id, ok := b.vocab[piece]; ok {
			return append(out, id)
		}
	}
	var stack [64]symbol
	symbols := stack[:0]
	for i := 0; i < len(piece); {
		r, w := utf8.DecodeRuneInString(piece[i:])
		var id int32
		if r < smallRunes && r != utf8.RuneError {
			id = b.small[r]
		} else if v, ok := b.vocab[piece[i:i+w]]; ok {
			id = v
		} else {
			id = -1
		}
		symbols = append(symbols, symbol{int32(i), int32(i + w), id})
		i += w
	}
	if len(symbols) >= heapMinSymbols {
		symbols = b.mergeWithHeap(symbols)
	} else {
		symbols = b.mergeByRescan(symbols)
	}
	return b.emit(piece, symbols, out)
}

// noMerge marks an adjacent pair no rule joins.
const noMerge = int32(1 << 30)

// mergeByRescan applies the lowest-ranked merge, leftmost on ties, until none applies. It keeps
// the rank of every adjacent pair and recomputes only the two a merge touches, so a piece costs
// one table lookup per merge instead of one per pair per merge.
func (b *bpe) mergeByRescan(symbols []symbol) []symbol {
	var stack [heapMinSymbols]mergeRule
	var rules []mergeRule
	if len(symbols) <= len(stack) {
		rules = stack[:len(symbols)]
	} else {
		rules = make([]mergeRule, len(symbols))
	}
	pair := func(i int) {
		if r, ok := b.rule(symbols[i], symbols[i+1]); ok {
			rules[i] = r
		} else {
			rules[i] = mergeRule{rank: noMerge}
		}
	}
	for i := 0; i+1 < len(symbols); i++ {
		pair(i)
	}
	for len(symbols) > 1 {
		bestRank, bestAt := noMerge, -1
		for i := 0; i+1 < len(symbols); i++ {
			if rules[i].rank < bestRank {
				bestRank, bestAt = rules[i].rank, i
			}
		}
		if bestAt < 0 {
			return symbols
		}
		symbols[bestAt] = symbol{symbols[bestAt].start, symbols[bestAt+1].end, rules[bestAt].id}
		symbols = append(symbols[:bestAt+1], symbols[bestAt+2:]...)
		rules = append(rules[:bestAt+1], rules[bestAt+2:]...)
		if bestAt+1 < len(symbols) {
			pair(bestAt)
		}
		if bestAt > 0 {
			pair(bestAt - 1)
		}
	}
	return symbols
}

type pairItem struct {
	rank, slot, version int32
}

type pairHeap []pairItem

func (h pairHeap) Len() int { return len(h) }
func (h pairHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].slot < h[j].slot
}
func (h pairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(x any)   { *h = append(*h, x.(pairItem)) }
func (h *pairHeap) Pop() any {
	old := *h
	item := old[len(old)-1]
	*h = old[:len(old)-1]
	return item
}

// mergeWithHeap is mergeByRescan for long pieces: a priority queue of adjacent pairs with
// versioned entries, so a stale pair is skipped instead of removed.
func (b *bpe) mergeWithHeap(symbols []symbol) []symbol {
	n := len(symbols)
	next := make([]int32, n)
	prev := make([]int32, n)
	version := make([]int32, n)
	dead := make([]bool, n)
	for i := range symbols {
		prev[i], next[i] = int32(i-1), int32(i+1)
	}
	next[n-1] = -1
	queue := make(pairHeap, 0, 3*n)
	offer := func(slot int32) {
		right := next[slot]
		if right < 0 {
			return
		}
		if r, ok := b.rule(symbols[slot], symbols[right]); ok {
			heap.Push(&queue, pairItem{r.rank, slot, version[slot]})
		}
	}
	for i := range symbols {
		offer(int32(i))
	}
	for queue.Len() > 0 {
		item := heap.Pop(&queue).(pairItem)
		slot := item.slot
		if dead[slot] || version[slot] != item.version {
			continue
		}
		right := next[slot]
		if right < 0 || dead[right] {
			continue
		}
		r, ok := b.rule(symbols[slot], symbols[right])
		if !ok {
			continue
		}
		symbols[slot] = symbol{symbols[slot].start, symbols[right].end, r.id}
		dead[right] = true
		next[slot] = next[right]
		if next[right] >= 0 {
			prev[next[right]] = slot
		}
		version[slot]++
		offer(slot)
		if left := prev[slot]; left >= 0 {
			version[left]++
			offer(left)
		}
	}
	out := symbols[:0]
	for i := int32(0); i >= 0 && int(i) < n; i = next[i] {
		if !dead[i] {
			out = append(out, symbols[i])
		}
	}
	return out
}

// emit turns merged symbols into ids. A symbol outside the vocabulary becomes its UTF-8 bytes
// when the checkpoint has byte fallback and every byte resolves, else the unknown token (one per
// run when fuse_unk is set), else it is dropped.
func (b *bpe) emit(piece string, symbols []symbol, out []int32) []int32 {
	unknownPending := false
	for _, s := range symbols {
		if s.id >= 0 {
			unknownPending = false
			out = append(out, s.id)
			continue
		}
		if b.byteIDs != nil {
			text := piece[s.start:s.end]
			resolves := true
			for i := 0; i < len(text); i++ {
				if b.byteIDs[text[i]] < 0 {
					resolves = false
					break
				}
			}
			if resolves {
				for i := 0; i < len(text); i++ {
					out = append(out, b.byteIDs[text[i]])
				}
				unknownPending = false
				continue
			}
		}
		if !b.hasUnk {
			continue
		}
		if b.fuseUnk && unknownPending {
			continue
		}
		out = append(out, b.unkID)
		unknownPending = true
	}
	return out
}
