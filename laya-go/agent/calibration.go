package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
	"github.com/NandhaKishorM/laya/laya-go/internal/unicodedata"
)

// calibrationField is the ConfigError.Field of a calibration payload that cannot be used.
const calibrationField = "calibration"

// CalibrationVersion is the payload version this package writes and reads as one that records
// the checkpoint it was fitted for. A payload with no version key is version 1.
const CalibrationVersion = 2

// configIdentitySkip are the config keys a calibration stores at its own top level.
var configIdentitySkip = map[string]bool{"temperature": true, "temperature_by_options": true}

// configIdentity is the checkpoint config without its temperature fields (_config_identity).
func configIdentity(cfg pyjson.Object) pyjson.Object {
	out := make(pyjson.Object, 0, len(cfg))
	for _, m := range cfg {
		if k, ok := m.Key.(string); ok && configIdentitySkip[k] {
			continue
		}
		out = append(out, m)
	}
	return out
}

// WithCalibration installs a calibration payload (laya's calibration JSON) at Open, as
// ONNXAgent's calibration argument does: the temperature table, the bucket temperatures and the
// binning map are replaced by the payload's, which is clamped to [0.5, 5], and language
// temperatures stay as WithLangTemperatures resolved them. A payload that is not the shape
// apply_calibration_payload reads makes Open fail with a *ConfigError whose Field is
// "calibration". A mismatch with the checkpoint or a clamped entry is a warning in
// Config().Warnings. An empty payload installs nothing.
func WithCalibration(payload []byte) Option {
	return func(o *options) { o.calibration = append([]byte(nil), payload...) }
}

// CalibrationPayload is the body laya writes for a fitted calibration (calibration_payload), for
// WithCalibration to read back.
type CalibrationPayload struct {
	// Temperature is one value per question type: choice, score, noul.
	Temperature [3]float64
	// TemperatureByOptions overrides Temperature per bucket ("choice:3-5").
	TemperatureByOptions map[string]float64
	// ModelIDOrPath and Subfolder name the checkpoint the map was fitted for; nil is JSON null.
	ModelIDOrPath, Subfolder *string
	// Config is the checkpoint's rl_agent_config.json; the temperature fields are not written.
	Config map[string]any
	// BinningMap is the optional recalibration map. nil omits the key; an empty map is written.
	BinningMap map[string]BinningEntry
}

// BinningEntry is one bucket of a binning map: Bins equal-width bins and one value in [0, 1] each.
type BinningEntry struct {
	Bins   int       `json:"bins"`
	Values []float64 `json:"values"`
}

// MarshalJSON writes the payload with version CalibrationVersion. A NaN or infinite temperature
// or binning value is an error, as encoding/json has no spelling for it, and so is a bucket whose
// Bins is not its number of values.
func (p CalibrationPayload) MarshalJSON() ([]byte, error) {
	cfg := map[string]any(nil)
	if p.Config != nil {
		cfg = make(map[string]any, len(p.Config))
		for k, v := range p.Config {
			if !configIdentitySkip[k] {
				cfg[k] = v
			}
		}
	}
	byOptions := p.TemperatureByOptions
	if byOptions == nil {
		byOptions = map[string]float64{}
	}
	for name, e := range p.BinningMap {
		if e.Bins < 1 || e.Bins != len(e.Values) {
			return nil, fmt.Errorf("binning_map[%q]: bins %d does not match %d values", name, e.Bins, len(e.Values))
		}
	}
	body := struct {
		Version     int                      `json:"version"`
		Temperature [3]float64               `json:"temperature"`
		ByOptions   map[string]float64       `json:"temperature_by_options"`
		Model       *string                  `json:"model_id_or_path"`
		Subfolder   *string                  `json:"subfolder"`
		Config      map[string]any           `json:"config"`
		BinningMap  *map[string]BinningEntry `json:"binning_map,omitempty"`
	}{CalibrationVersion, p.Temperature, byOptions, p.ModelIDOrPath, p.Subfolder, cfg, nil}
	if p.BinningMap != nil {
		body.BinningMap = &p.BinningMap
	}
	return json.Marshal(body)
}

// applyCalibration is apply_calibration_payload for an ONNXAgent: it checks the payload's shape,
// warns on an identity mismatch (version 2 and later) and on clamped temperatures, and returns
// the tables with the payload's installed.
func applyCalibration(cfg *Config, t *decode.Temperatures, data []byte) (*decode.Temperatures, error) {
	v, err := pyvalue.Parse(data)
	if err != nil {
		return nil, err
	}
	payload, ok := v.(pyjson.Object)
	if !ok {
		return nil, fmt.Errorf("calibration JSON must be an object, got %s", pyvalue.TypeName(v))
	}
	checkIdentity, err := versionChecksIdentity(objGet(payload, "version"))
	if err != nil {
		return nil, err
	}
	temps, ok := objGet(payload, "temperature").([]any)
	if !ok || len(temps) != 3 {
		return nil, errors.New("calibration JSON must contain temperature: [3 floats]")
	}
	var byOptions pyjson.Object
	switch raw := objGet(payload, "temperature_by_options").(type) {
	case nil: // absent, or an explicit null: a file from before bucket temperatures
	case pyjson.Object:
		byOptions = raw
	default:
		return nil, fmt.Errorf("calibration JSON temperature_by_options must be an object of bucket -> float, got %s", pyvalue.TypeName(raw))
	}
	binning, err := readBinning(objGet(payload, "binning_map"))
	if err != nil {
		return nil, err
	}
	if checkIdentity {
		if w := identityMismatch(cfg.identity, payload); w != "" {
			cfg.Warnings = append(cfg.Warnings, w)
		}
	}
	raw := make(map[string]any, len(byOptions))
	for _, m := range byOptions {
		raw[m.Key.(string)] = m.Value
	}
	out := t.Calibrated(temps, raw, binning)
	var rejected []string
	note := func(name string, raw any, applied float64) {
		if !sameNumber(raw, applied) {
			rejected = append(rejected, fmt.Sprintf("%s=%s -> %s", name, pyvalue.Repr(raw),
				strconv.FormatFloat(applied, 'g', -1, 64)))
		}
	}
	for i, r := range temps {
		note(fmt.Sprintf("temperature[%d]", i), r, out.ByType[i])
	}
	for _, m := range byOptions {
		note(m.Key.(string), m.Value, out.ByOptions[m.Key.(string)])
	}
	if len(rejected) > 0 {
		cfg.Warnings = append(cfg.Warnings, fmt.Sprintf(
			"laya: calibration temperatures are invalid or outside [%g, %g]; using %s. "+
				"Treat confidence from the affected entries as uncalibrated.",
			decode.TempMin, decode.TempMax, strings.Join(rejected, ", ")))
	}
	return out, nil
}

func objGet(o pyjson.Object, key string) any {
	for _, m := range o {
		if m.Key == key {
			return m.Value
		}
	}
	return nil
}

// versionChecksIdentity reports whether the payload's version is CalibrationVersion or later.
// None and an absent key are version 1; a bool, a list or an object are refused, and int() of a
// float must be whole, of a string reads an integer literal.
func versionChecksIdentity(v any) (bool, error) {
	bad := func() (bool, error) {
		return false, fmt.Errorf("calibration JSON version must be an integer, got %s", pyvalue.Repr(v))
	}
	switch x := v.(type) {
	case nil:
		return false, nil
	case json.Number:
		if !strings.ContainsAny(string(x), ".eE") {
			n, ok := new(big.Int).SetString(string(x), 10)
			if !ok {
				return bad()
			}
			return n.Cmp(big.NewInt(CalibrationVersion)) >= 0, nil
		}
		f, _ := strconv.ParseFloat(string(x), 64)
		if math.IsInf(f, 0) || f != math.Trunc(f) {
			return bad()
		}
		return f >= CalibrationVersion, nil
	case float64: // the NaN and Infinity literals
		return bad()
	case string:
		n, ok := pyInt(x)
		if !ok {
			return bad()
		}
		return n.Cmp(big.NewInt(CalibrationVersion)) >= 0, nil
	}
	return bad()
}

// pyInt is int(s) for an integer literal: surrounding space, a sign, and Unicode decimal digits
// with single underscores between them.
func pyInt(s string) (*big.Int, bool) {
	s = unicodedata.DecimalASCII(unicodedata.StripNumeric(s))
	neg := false
	if s != "" && (s[0] == '+' || s[0] == '-') {
		neg = s[0] == '-'
		s = s[1:]
	}
	if s == "" || s[0] == '_' || s[len(s)-1] == '_' || strings.Contains(s, "__") {
		return nil, false
	}
	s = strings.ReplaceAll(s, "_", "")
	for _, c := range s {
		if c < '0' || c > '9' {
			return nil, false
		}
	}
	if len(s) > pyvalue.MaxIntDigits {
		return nil, false
	}
	n, ok := new(big.Int).SetString(s, 10)
	if ok && neg {
		n.Neg(n)
	}
	return n, ok
}

// readBinning checks a payload's binning_map and returns it as bins; nil for none. The map is
// refused, not clamped, when any value is not a finite number in [0, 1].
func readBinning(v any) (map[string]decode.Bins, error) {
	if v == nil {
		return nil, nil
	}
	m, ok := v.(pyjson.Object)
	if !ok {
		return nil, fmt.Errorf("calibration JSON binning_map must be an object of bucket -> {bins, values}, got %s", pyvalue.TypeName(v))
	}
	out := make(map[string]decode.Bins, len(m))
	for _, member := range m {
		name := member.Key.(string)
		entry, ok := member.Value.(pyjson.Object)
		if !ok {
			return nil, fmt.Errorf("calibration JSON binning_map[%s] must be an object with \"bins\" and \"values\", got %s",
				pyvalue.Repr(name), pyvalue.TypeName(member.Value))
		}
		rawBins, rawValues := objGet(entry, "bins"), objGet(entry, "values")
		n, isInt := rawBins.(json.Number)
		bins, ok := new(big.Int).SetString(string(n), 10)
		if !isInt || !ok || bins.Sign() < 1 {
			return nil, fmt.Errorf("calibration JSON binning_map[%s] must have an integer \"bins\" >= 1, got %s", pyvalue.Repr(name), pyvalue.Repr(rawBins))
		}
		list, isList := rawValues.([]any)
		if !isList || !bins.IsInt64() || bins.Int64() != int64(len(list)) {
			return nil, fmt.Errorf("calibration JSON binning_map[%s] must have \"values\" of length \"bins\" (%s)", pyvalue.Repr(name), bins)
		}
		values := make([]float64, len(list))
		for i, x := range list {
			f, ok := numberValue(x)
			if !ok {
				return nil, fmt.Errorf("calibration JSON binning_map[%s] values must be numbers, got %s", pyvalue.Repr(name), pyvalue.Repr(rawValues))
			}
			values[i] = f
		}
		for _, f := range values {
			if math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("calibration JSON binning_map[%s] values must be finite, got %s", pyvalue.Repr(name), pyvalue.Repr(rawValues))
			}
			if f < 0 || f > 1 {
				return nil, fmt.Errorf("calibration JSON binning_map[%s] values must be in [0, 1], got %s", pyvalue.Repr(name), pyvalue.Repr(rawValues))
			}
		}
		out[name] = decode.Bins{Values: values}
	}
	return out, nil
}

// numberValue is a JSON int or float (not a bool, not a string) as a float64; a literal too large
// for one is infinite, as Python's float() of it is.
func numberValue(v any) (float64, bool) {
	switch x := v.(type) {
	case json.Number:
		f, err := strconv.ParseFloat(string(x), 64)
		var numErr *strconv.NumError
		if err != nil && !(errors.As(err, &numErr) && numErr.Err == strconv.ErrRange) {
			return 0, false
		}
		return f, true
	case float64:
		return x, true
	}
	return 0, false
}

// identityMismatch is _warn_if_identity_mismatch for an agent that records no model id or
// subfolder: the payload must name neither, and carry exactly the checkpoint's config.
func identityMismatch(current, payload pyjson.Object) string {
	model, sub, recorded := objGet(payload, "model_id_or_path"), objGet(payload, "subfolder"), objGet(payload, "config")
	same := pyvalue.Equal(recorded, current)
	if model == nil && sub == nil && same {
		return ""
	}
	note := ""
	if !same {
		note = " Checkpoint config identity differs."
	}
	return fmt.Sprintf("calibration was fitted for model_id_or_path=%s subfolder=%s but this agent is "+
		"model_id_or_path=None subfolder=None.%s Loading the temperatures anyway.",
		pyvalue.Repr(model), pyvalue.Repr(sub), note)
}
