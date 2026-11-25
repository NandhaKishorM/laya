package decode

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
	"github.com/NandhaKishorM/laya/laya-go/schema"
)

// The bounds clamp applies: a fitted temperature below 0.5 sharpens a coin flip into a
// certainty (the shipped choice:11+ bucket is 0.1006), so it is refused.
const (
	TempMin = 0.5
	TempMax = 5.0
)

// Temperatures are a checkpoint's temperature tables, every value already clamped.
type Temperatures struct {
	// ByType is the temperature per question type, indexed by Type.Code.
	ByType [3]float64
	// ByOptions overrides ByType per bucket ("choice:3-5"; see Bucket).
	ByOptions map[string]float64
	// ByLanguage overrides both per primary language subtag ("zh").
	ByLanguage map[string]LanguageTemperatures
	// Binning recalibrates AnswerConfidence per bucket; nil for none.
	Binning map[string]Bins
}

// LanguageTemperatures is one language's override, resolved as resolve_lang_temperatures does:
// a missing ByType inherits the checkpoint's raw (unclamped, then clamped) table.
type LanguageTemperatures struct {
	ByType    [3]float64
	ByOptions map[string]float64
}

// Bins is one bucket of a histogram-binning map: a confidence c maps to
// Values[min(len-1, max(0, int(c*len)))].
type Bins struct {
	Values []float64
}

// NewTemperatures resolves the raw config values (temperature, temperature_by_options) and the
// raw lang_temperatures option into clamped tables. Raw values are any, as JSON decodes them,
// because a bool or a non-number clamps to 1.0 rather than failing.
//
// byType is nil (three 1.0) or three values. byLanguage maps a language tag to
// {"temperature": [3 values] or null, "temperature_by_options": {bucket: value}} or to nil, each
// as a map[string]any; keys are reduced to their primary subtag, lower-cased. The result is
// immutable by convention: it is only read, so it may be shared between goroutines.
func NewTemperatures(byType []any, byOptions map[string]any, byLanguage map[string]any) (*Temperatures, error) {
	t := &Temperatures{ByOptions: map[string]float64{}}
	base := byType
	if base == nil {
		base = []any{1.0, 1.0, 1.0}
	}
	if len(base) != 3 {
		return nil, fmt.Errorf("decode: temperature must list 3 values, one per question type, got %d", len(base))
	}
	for i, v := range base {
		t.ByType[i] = Clamp(v)
	}
	for bucket, v := range byOptions {
		t.ByOptions[bucket] = Clamp(v)
	}
	if len(byLanguage) > 0 {
		t.ByLanguage = make(map[string]LanguageTemperatures, len(byLanguage))
	}
	langs := make([]string, 0, len(byLanguage))
	for lang := range byLanguage {
		langs = append(langs, lang)
	}
	seen := map[string]string{}
	sort.Strings(langs) // a repeated primary subtag resolves the same way on every run
	for _, lang := range langs {
		resolved, err := resolveLanguage(lang, byLanguage[lang], base)
		if err != nil {
			return nil, err
		}
		p := primarySubtag(lang)
		if first, dup := seen[p]; dup {
			return nil, fmt.Errorf("decode: language overrides %q and %q share the primary subtag %q; Python keeps the last inserted, which a Go map cannot say, so give one", first, lang, p)
		}
		seen[p] = lang
		t.ByLanguage[primarySubtag(lang)] = resolved
	}
	return t, nil
}

// Calibrated is t with the type and bucket tables and the binning map replaced, as
// apply_calibration_payload does: language overrides stay as Open resolved them. byType holds
// three raw values; every value is clamped.
func (t *Temperatures) Calibrated(byType []any, byOptions map[string]any, binning map[string]Bins) *Temperatures {
	out := &Temperatures{ByOptions: make(map[string]float64, len(byOptions)), Binning: binning}
	if t != nil {
		out.ByLanguage = t.ByLanguage
	}
	for i, v := range byType {
		out.ByType[i] = Clamp(v)
	}
	for bucket, v := range byOptions {
		out.ByOptions[bucket] = Clamp(v)
	}
	return out
}

// primarySubtag is lang.split("-")[0].lower().
func primarySubtag(lang string) string {
	primary, _, _ := strings.Cut(lang, "-")
	return unicodedata.Lower(primary)
}

func resolveLanguage(lang string, raw any, base []any) (LanguageTemperatures, error) {
	var cfg map[string]any
	switch x := raw.(type) {
	case nil:
	case map[string]any:
		cfg = x
	default:
		return LanguageTemperatures{}, fmt.Errorf("decode: language override %q must be a mapping, got %T", lang, raw)
	}
	list := base
	if tr := cfg["temperature"]; tr != nil { // null inherits the checkpoint's own
		var ok bool
		if list, ok = asList(tr); !ok || len(list) != 3 {
			return LanguageTemperatures{}, fmt.Errorf("decode: language override %q temperature must be a list of 3 floats, got %v", lang, tr)
		}
	}
	out := LanguageTemperatures{ByOptions: map[string]float64{}}
	for i, v := range list {
		out.ByType[i] = Clamp(v)
	}
	if bo := cfg["temperature_by_options"]; !falsy(bo) { // `or {}`: false, 0, "", [] and {} are empty
		m, ok := asMap(bo)
		if !ok {
			return LanguageTemperatures{}, fmt.Errorf("decode: language override %q temperature_by_options must be a mapping of bucket -> float, got %T", lang, bo)
		}
		for bucket, v := range m {
			out.ByOptions[bucket] = Clamp(v)
		}
	}
	return out, nil
}

func asList(v any) ([]any, bool) {
	if l, ok := v.([]any); ok {
		return l, true
	}
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || (rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array) || rv.Type().Elem().Kind() == reflect.Uint8 {
		return nil, false
	}
	out := make([]any, rv.Len())
	for i := range out {
		out[i] = rv.Index(i).Interface()
	}
	return out, true
}

func asMap(v any) (map[string]any, bool) {
	if m, ok := v.(map[string]any); ok {
		return m, true
	}
	rv := reflect.ValueOf(v)
	if !rv.IsValid() || rv.Kind() != reflect.Map || rv.Type().Key().Kind() != reflect.String {
		return nil, false
	}
	out := make(map[string]any, rv.Len())
	for _, k := range rv.MapKeys() {
		out[k.String()] = rv.MapIndex(k).Interface()
	}
	return out, true
}

// For is the temperature for a question of type typ with k options, and whether a language
// override chose it. A nil Temperatures is all 1.0.
func (t *Temperatures) For(typ schema.Type, k int, lang string) (temperature float64, override bool) {
	if t == nil || !typ.Valid() {
		return 1, false
	}
	bucket := Bucket(typ, k)
	temperature, ok := t.ByOptions[bucket]
	if !ok {
		temperature = t.ByType[typ.Code()]
	}
	if lang != "" {
		if l, ok := t.ByLanguage[primarySubtag(lang)]; ok {
			temperature, ok = l.ByOptions[bucket]
			if !ok {
				temperature = l.ByType[typ.Code()]
			}
			return temperature, true
		}
	}
	return temperature, false
}

// Bucket is laya's temp_bucket: "<type>:2", ":3-5", ":6-10" or ":11+" by option count.
func Bucket(typ schema.Type, k int) string {
	switch {
	case k <= 2:
		return string(typ) + ":2"
	case k <= 5:
		return string(typ) + ":3-5"
	case k <= 10:
		return string(typ) + ":6-10"
	}
	return string(typ) + ":11+"
}

// Clamp is laya's clamp_temperature: v confined to [TempMin, TempMax], or 1.0 when v is a bool,
// not a number, NaN or infinite. A string is read as float() reads it, without the underscore
// and non-ASCII digit spellings.
func Clamp(v any) float64 {
	f, ok := toFloat(v)
	if !ok || math.IsNaN(f) || math.IsInf(f, 0) {
		return 1
	}
	return math.Min(TempMax, math.Max(TempMin, f))
}

func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case bool:
		return 0, false
	case float64:
		return x, true
	case float32:
		return float64(x), true
	case json.Number:
		f, err := strconv.ParseFloat(string(x), 64)
		return f, err == nil || isRange(err)
	case *big.Int:
		f, _ := new(big.Float).SetInt(x).Float64()
		return f, true
	case string:
		return PyFloat(x)
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(rv.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(rv.Uint()), true
	}
	return 0, false
}

// PyFloat is float(s) for a string: surrounding space, Unicode decimal digits, single underscores
// between digits, and a signed nan, inf or infinity in any case. Go's hex floats are refused. A
// value past the float range is infinite, as in Python.
func PyFloat(s string) (float64, bool) {
	s = unicodedata.DecimalASCII(unicodedata.StripNumeric(s))
	if strings.ContainsAny(s, "xXpP") {
		return 0, false
	}
	if strings.Contains(s, "_") {
		var b strings.Builder
		for i := 0; i < len(s); i++ {
			if s[i] == '_' {
				if i == 0 || i+1 == len(s) || !isDigit(s[i-1]) || !isDigit(s[i+1]) {
					return 0, false
				}
				continue
			}
			b.WriteByte(s[i])
		}
		s = b.String()
	}
	if u := strings.TrimLeft(s, "+-"); len(s)-len(u) == 1 && strings.EqualFold(u, "nan") {
		return math.NaN(), true
	}
	f, err := strconv.ParseFloat(s, 64)
	return f, err == nil || isRange(err)
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func isRange(err error) bool {
	ne, ok := err.(*strconv.NumError)
	return ok && ne.Err == strconv.ErrRange
}

// falsy is Python truthiness for the values `cfg.get(...) or {}` meets: None, False, zero, "" and
// an empty container.
func falsy(v any) bool {
	if n, ok := v.(json.Number); ok { // a string kind: checked before the reflect switch
		f, err := n.Float64()
		return err == nil && f == 0
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Invalid:
		return true
	case reflect.Bool:
		return !rv.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return rv.Float() == 0
	case reflect.String, reflect.Slice, reflect.Array, reflect.Map:
		return rv.Len() == 0
	}
	return false
}
