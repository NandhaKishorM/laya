package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ErrUnsupported is wrapped by every error that refuses a tokenizer feature this package does
// not implement. Test for it with errors.Is.
var ErrUnsupported = errors.New("unsupported tokenizer component")

func unsupported(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrUnsupported}, args...)...)
}

// node is a decoded JSON object from a tokenizer.json component section.
type node map[string]any

// parseNode decodes a component section; JSON null gives a nil node.
func parseNode(raw json.RawMessage, what string) (node, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var n node
	if err := json.Unmarshal(raw, &n); err != nil {
		return nil, fmt.Errorf("%s must be an object or null: %w", what, err)
	}
	return n, nil
}

func asNode(v any, what string) (node, error) {
	n, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be an object", what)
	}
	return n, nil
}

func (n node) str(key, def string) (string, error) {
	v, ok := n[key]
	if !ok || v == nil {
		return def, nil
	}
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("%q must be a string, found %T", key, v)
	}
	return s, nil
}

func (n node) boolean(key string, def bool) (bool, error) {
	v, ok := n[key]
	if !ok || v == nil {
		return def, nil
	}
	b, ok := v.(bool)
	if !ok {
		return false, fmt.Errorf("%q must be a boolean, found %T", key, v)
	}
	return b, nil
}

func (n node) integer(key string, def int) (int, error) {
	v, ok := n[key]
	if !ok || v == nil {
		return def, nil
	}
	f, ok := v.(float64)
	if !ok || f != float64(int(f)) {
		return 0, fmt.Errorf("%q must be an integer, found %v", key, v)
	}
	return int(f), nil
}

func (n node) list(key string) ([]any, error) {
	l, ok := n[key].([]any)
	if !ok {
		return nil, fmt.Errorf("%q must be an array", key)
	}
	return l, nil
}

func (n node) object(key string) (node, error) {
	return asNode(n[key], fmt.Sprintf("%q", key))
}
