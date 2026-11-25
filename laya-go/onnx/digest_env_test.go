package onnx

import (
	"errors"
	"strings"
	"testing"
)

func TestVerifyGraphFromEnv(t *testing.T) {
	path, sum := stubGraph(t)
	zero := strings.Repeat("0", 64)
	for name, c := range map[string]struct {
		env string
		ok  bool
	}{
		"onnx ok":          {`{"onnx":"` + sum + `"}`, true},
		"onnx_path ok":     {`{"onnx_path":"` + sum + `"}`, true},
		"onnx bad":         {`{"onnx":"` + zero + `"}`, false},
		"onnx_path bad":    {`{"onnx_path":"` + zero + `"}`, false},
		"both equal":       {`{"onnx":"` + sum + `","onnx_path":"` + strings.ToUpper(sum) + `"}`, true},
		"both differ":      {`{"onnx":"` + sum + `","onnx_path":"` + zero + `"}`, false},
		"other keys only":  {`{"rl_agent_config.json":"` + zero + `"}`, true},
		"unset":            {"", true},
		"not json":         {"nope", false},
		"non-string value": {`{"onnx":1}`, false},
	} {
		t.Run(name, func(t *testing.T) {
			t.Setenv("LAYA_SHA256_DIGESTS", c.env)
			if err := verifyGraph(path, options{}); c.ok != (err == nil) {
				t.Fatalf("err = %v", err)
			}
		})
	}
	t.Setenv("LAYA_SHA256_DIGESTS", `{"onnx":"`+zero+`"}`)
	var o options
	WithExpectedSHA256(sum)(&o)
	if err := verifyGraph(path, o); err != nil {
		t.Fatalf("an explicit option wins: %v", err)
	}
	if err := verifyGraph(path, options{}); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("env mismatch = %v", err)
	}
}
