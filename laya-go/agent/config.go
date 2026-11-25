package agent

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"strconv"
	"strings"

	"github.com/NandhaKishorM/laya/laya-go/internal/decode"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyjson"
	"github.com/NandhaKishorM/laya/laya-go/internal/pyvalue"
)

// configFile is the checkpoint file the configuration is read from.
const configFile = "rl_agent_config.json"

// The defaults ONNXAgent applies to a config that does not name them.
const (
	defaultMaxLen     = 512
	defaultHeadMaxLen = 192
	layoutSequential  = "sequential"
)

// ErrUnsupportedLayout is wrapped when a checkpoint's option_layout is not "sequential", the
// only layout the exported graph takes.
var ErrUnsupportedLayout = errors.New("agent: option_layout is not supported by the ONNX runtime")

// ConfigError reports a missing or malformed rl_agent_config.json.
type ConfigError struct {
	// Field is the config key at fault, or "" for the file as a whole.
	Field string
	Err   error
}

// tokenizerField is the ConfigError.Field of a tokenizer that could not be loaded.
const tokenizerField = "tokenizer"

// Error names the file, the field when there is one, and the cause.
func (e *ConfigError) Error() string {
	cause := "<nil>"
	if e.Err != nil {
		cause = e.Err.Error()
	}
	// One prefix: a cause may carry its own package's ("agent: " from a sentinel, "decode: ").
	cause = strings.TrimPrefix(strings.TrimPrefix(cause, "agent: "), "decode: ")
	switch e.Field {
	case "":
		return "agent: " + configFile + ": " + cause
	case tokenizerField:
		return "agent: tokenizer: " + cause // not a field of the config file
	}
	return "agent: " + configFile + ": " + e.Field + ": " + cause
}

// Unwrap returns the cause.
func (e *ConfigError) Unwrap() error { return e.Err }

// Config is the checkpoint configuration an Agent runs with. Temperatures are not in it; the
// Agent holds them clamped.
type Config struct {
	// MaxLen is the whole sequence budget: max_len, default 512.
	MaxLen int
	// HeadMaxLen is the question half of it: head_max_len, default 192.
	HeadMaxLen int
	// OptionLayout is option_layout, "sequential" by default and the only one accepted.
	OptionLayout string
	// Warnings are what loading changed without failing, such as temperatures outside
	// [0.5, 5] that were clamped.
	Warnings []string

	identity pyjson.Object // the file without its temperature fields, for a calibration's check
}

// LoadConfig reads and checks rl_agent_config.json from the root of checkpoint, as ONNXAgent
// does: max_len, head_max_len and option_layout, and a warning for every temperature clamped.
func LoadConfig(checkpoint fs.FS) (Config, error) {
	cfg, _, err := loadConfig(checkpoint, nil)
	return cfg, err
}

func loadConfig(checkpoint fs.FS, langs map[string]any) (Config, *decode.Temperatures, error) {
	data, err := fs.ReadFile(checkpoint, configFile)
	if err != nil {
		return Config{}, nil, &ConfigError{Err: err}
	}
	v, err := pyvalue.Parse(data)
	if err != nil {
		return Config{}, nil, &ConfigError{Err: err}
	}
	obj, ok := v.(pyjson.Object)
	if !ok {
		return Config{}, nil, &ConfigError{Err: fmt.Errorf("want a JSON object, got %s", pyvalue.TypeName(v))}
	}
	get := func(key string) (any, bool) {
		for _, m := range obj {
			if m.Key == key {
				return m.Value, true
			}
		}
		return nil, false
	}
	cfg := Config{MaxLen: defaultMaxLen, HeadMaxLen: defaultHeadMaxLen, OptionLayout: layoutSequential,
		identity: configIdentity(obj)}
	if raw, ok := get("option_layout"); ok {
		if layout, isString := raw.(string); !isString || layout != layoutSequential {
			return Config{}, nil, &ConfigError{Field: "option_layout",
				Err: fmt.Errorf("%w: %s", ErrUnsupportedLayout, pyvalue.Repr(raw))}
		}
	}
	for _, f := range []struct {
		key string
		to  *int
	}{{"max_len", &cfg.MaxLen}, {"head_max_len", &cfg.HeadMaxLen}} {
		raw, ok := get(f.key)
		if !ok {
			continue
		}
		n, isNumber := raw.(json.Number)
		size, err := strconv.Atoi(string(n))
		var numErr *strconv.NumError
		if errors.As(err, &numErr) && numErr.Err == strconv.ErrRange && !strings.HasPrefix(string(n), "-") {
			return Config{}, nil, &ConfigError{Field: f.key,
				Err: fmt.Errorf("%s is too large for an int", n)}
		}
		if !isNumber || err != nil || size <= 0 {
			return Config{}, nil, &ConfigError{Field: f.key,
				Err: fmt.Errorf("want a positive integer, got %s", pyvalue.Repr(raw))}
		}
		*f.to = size
	}

	var byType []any
	if raw, ok := get("temperature"); ok {
		if byType, ok = raw.([]any); !ok {
			return Config{}, nil, &ConfigError{Field: "temperature",
				Err: fmt.Errorf("want a list of 3 numbers, got %s", pyvalue.Repr(raw))}
		}
	}
	var byOptions map[string]any
	var optionKeys []string
	if raw, ok := get("temperature_by_options"); ok {
		members, isObject := raw.(pyjson.Object)
		if !isObject {
			return Config{}, nil, &ConfigError{Field: "temperature_by_options",
				Err: fmt.Errorf("want an object of bucket to number, got %s", pyvalue.Repr(raw))}
		}
		byOptions = make(map[string]any, len(members))
		for _, m := range members {
			key := m.Key.(string)
			byOptions[key] = m.Value
			optionKeys = append(optionKeys, key)
		}
	}
	temps, err := decode.NewTemperatures(byType, byOptions, nil)
	if err != nil {
		return Config{}, nil, &ConfigError{Field: "temperature", Err: err}
	}
	if len(langs) > 0 { // resolved against the raw base, as resolve_lang_temperatures is
		if temps, err = decode.NewTemperatures(byType, byOptions, langs); err != nil {
			return Config{}, nil, &ConfigError{Field: langTempsField, Err: err}
		}
	}

	// ONNXAgent warns once about every entry the clamp changed, buckets first.
	var rejected []string
	note := func(name string, raw any, applied float64) {
		if !sameNumber(raw, applied) {
			rejected = append(rejected, fmt.Sprintf("%s=%s -> %s", name, pyvalue.Repr(raw),
				strconv.FormatFloat(applied, 'g', -1, 64)))
		}
	}
	for _, key := range optionKeys {
		note(key, byOptions[key], temps.ByOptions[key])
	}
	for i, raw := range byType {
		note(fmt.Sprintf("temperature[%d]", i), raw, temps.ByType[i])
	}
	if len(rejected) > 0 {
		cfg.Warnings = append(cfg.Warnings, fmt.Sprintf(
			"this checkpoint ships temperatures outside [%g, %g] which would distort confidence; clamping %s. "+
				"Treat confidence from the affected buckets as uncalibrated.",
			decode.TempMin, decode.TempMax, strings.Join(rejected, ", ")))
	}
	return cfg, temps, nil
}

// sameNumber is float(raw) == applied: false for anything float() refuses. A string is read as
// by decode.PyFloat. A bool is never the applied number.
func sameNumber(raw any, applied float64) bool {
	switch x := raw.(type) {
	case string:
		f, ok := decode.PyFloat(x)
		return ok && f == applied
	case json.Number:
		f, err := strconv.ParseFloat(string(x), 64)
		return err == nil && f == applied
	}
	return false // a bool is no number to clamp_temperature
}
