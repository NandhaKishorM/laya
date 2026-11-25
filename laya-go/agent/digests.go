package agent

import (
	"io/fs"
	"sort"

	"github.com/NandhaKishorM/laya/laya-go/internal/digest"
)

// langTempsField is the ConfigError.Field of a lang_temperatures option that cannot be used.
const langTempsField = "lang_temperatures"

// ErrDigestMismatch is wrapped by the Open error for a file whose SHA-256 is not the expected one.
var ErrDigestMismatch = digest.ErrMismatch

// ErrUnsafeDigestPath is wrapped by the Open error for a WithExpectedSHA256 name that is absolute
// or leaves the checkpoint.
var ErrUnsafeDigestPath = digest.ErrUnsafePath

// WithLangTemperatures sets per-language temperatures, in the raw shape of laya's
// lang_temperatures option: language tag -> {"temperature": [3 numbers] or nil,
// "temperature_by_options": {bucket: number}}, an entry or its temperature being nil to inherit the
// checkpoint's own. Tags are reduced to their primary subtag, lower-cased. A call that names a
// language through WithLanguage then decodes with that language's table. A malformed entry makes
// Open fail with a *ConfigError whose Field is "lang_temperatures". Values are clamped to [0.5, 5].
func WithLangTemperatures(langs map[string]map[string]any) Option {
	return func(o *options) {
		o.langTemps = make(map[string]any, len(langs))
		for lang, cfg := range langs {
			o.langTemps[lang] = cfg // a nil map inherits the checkpoint's own, as None does
		}
	}
}

// WithExpectedSHA256 makes Open check files of the checkpoint against hex SHA-256 digests, keyed
// by slash path under it, before any of them is parsed. A file that is absent or a directory fails
// Open with an error wrapping fs.ErrNotExist, a different digest with one wrapping
// ErrDigestMismatch that names the file and both digests, and a name that is absolute or climbs
// out with ".." with one wrapping ErrUnsafeDigestPath. Files are hashed as a stream. To check the
// graph too, give onnx.Open onnx.WithExpectedSHA256. Without this option Open reads
// LAYA_SHA256_DIGESTS, a JSON object of name -> digest, as Python does (a value that is not valid
// JSON, not an object or not a string is a *ConfigError on that field); its "onnx" and "onnx_path"
// keys are left to onnx.Open. Giving the option, even empty, ignores the variable.
func WithExpectedSHA256(digests map[string]string) Option {
	return func(o *options) {
		o.sha256 = make(map[string]string, len(digests))
		for k, v := range digests {
			o.sha256[k] = v
		}
	}
}

// expectedDigests is what to verify: the option when WithExpectedSHA256 was given, else
// LAYA_SHA256_DIGESTS, as Python does, without the "onnx" and "onnx_path" keys, which name the
// graph and are checked by onnx.Open. A bad variable is a *ConfigError.
func expectedDigests(given map[string]string) (map[string]string, error) {
	if given != nil {
		return given, nil
	}
	env, err := digest.FromEnv()
	if err != nil {
		return nil, &ConfigError{Field: digest.Env, Err: err}
	}
	delete(env, "onnx")
	delete(env, "onnx_path")
	return env, nil
}

func verifyDigests(checkpoint fs.FS, want map[string]string) error {
	names := make([]string, 0, len(want))
	for name := range want {
		names = append(names, name)
	}
	sort.Strings(names) // the same file is named first on every run
	for _, name := range names {
		if err := digest.File(checkpoint, name, want[name]); err != nil {
			return err
		}
	}
	return nil
}
