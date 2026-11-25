package onnx

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/NandhaKishorM/laya/laya-go/infer"
)

func goodInputs() []tensorInfo {
	return []tensorInfo{
		{"input_ids", "int64", []int64{-1, -1}}, {"attention_mask", "int64", []int64{-1, -1}},
		{"marker_pos", "int64", []int64{-1, -1}}, {"marker_mask", "bool", []int64{-1, -1}},
		{"qtype", "int64", []int64{-1}},
	}
}

func goodOutputs() []tensorInfo {
	return []tensorInfo{{"logits", "float32", []int64{-1, -1}}, {"act_logits", "float32", []int64{-1, 2}}}
}

func TestCheckGraph(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(in, out []tensorInfo) ([]tensorInfo, []tensorInfo)
		problem string // "" for accepted
	}{
		{"current export", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { return i, o }, ""},
		{"reordered inputs", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[0], i[4] = i[4], i[0]; return i, o }, ""},
		{"act_logits fully dynamic", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { o[1].Dims = []int64{-1, -1}; return i, o }, ""},
		{"fixed batch on act_logits", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { o[1].Dims = []int64{1, 2}; return i, o }, `"act_logits" declares a fixed batch dimension 1`},
		{"fixed batch on an input", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[2].Dims = []int64{2, -1}; return i, o }, `"marker_pos" declares a fixed batch dimension 2`},
		{"fixed batch on qtype", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[4].Dims = []int64{1}; return i, o }, `"qtype" declares a fixed batch`},
		{"fixed sequence length", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[0].Dims = []int64{-1, 17}; return i, o }, `"input_ids" declares a fixed size 17`},
		{"fixed marker count", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { o[0].Dims = []int64{-1, 3}; return i, o }, `"logits" declares a fixed size 3`},
		{"act_logits width other than 2", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { o[1].Dims = []int64{-1, 3}; return i, o }, `"act_logits" declares a fixed size 3`},
		{"input_ids width of 2 is not act_logits", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[0].Dims = []int64{-1, 2}; return i, o }, `"input_ids" declares a fixed size 2`},
		{"missing input", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { return i[:4], o }, `missing input "qtype"`},
		{"missing output", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { return i, o[:1] }, `missing output "act_logits"`},
		{"renamed input", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[1].Name = "mask"; return i, o }, `missing input "attention_mask"`},
		{"extra input", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) {
			return append(i, tensorInfo{"token_type_ids", "int64", []int64{-1, -1}}), o
		}, `unexpected input "token_type_ids"`},
		{"extra output", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) {
			return i, append(o, tensorInfo{"hidden", "float32", []int64{-1, -1}})
		}, `unexpected output "hidden"`},
		{"int32 ids", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[0].DType = "int32"; return i, o }, `input "input_ids" is int32, want int64`},
		{"int64 marker_mask", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[3].DType = "int64"; return i, o }, `input "marker_mask" is int64, want bool`},
		{"float16 logits", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { o[0].DType = "float16"; return i, o }, `output "logits" is float16, want float32`},
		{"rank 3 ids", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[0].Dims = []int64{-1, -1, -1}; return i, o }, `input "input_ids" has rank 3, want 2`},
		{"qtype as [batch,1]", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { i[4].Dims = []int64{-1, 1}; return i, o }, `input "qtype" has rank 2, want 1`},
		{"scalar act_logits", func(i, o []tensorInfo) ([]tensorInfo, []tensorInfo) { o[1].Dims = nil; return i, o }, `output "act_logits" has rank 0, want 2`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := checkGraph(c.mutate(goodInputs(), goodOutputs()))
			if c.problem == "" {
				if err != nil {
					t.Fatalf("checkGraph = %v", err)
				}
				return
			}
			var ge *GraphError
			if !errors.As(err, &ge) || !strings.Contains(ge.Problem, c.problem) {
				t.Fatalf("checkGraph = %v, want a *GraphError containing %q", err, c.problem)
			}
			if !strings.Contains(ge.Problem, "scripts/export_onnx.py") {
				t.Errorf("no re-export hint in %q", ge.Problem)
			}
		})
	}
}

func TestGraphErrorMessage(t *testing.T) {
	err := &GraphError{Path: "a/laya.onnx", Problem: "broken"}
	if got := err.Error(); got != "onnx: graph a/laya.onnx: broken" {
		t.Fatalf("Error = %q", got)
	}
}

func TestCheckPath(t *testing.T) {
	dir := t.TempDir()
	touch := func(parts ...string) string {
		p := filepath.Join(append([]string{dir}, parts...)...)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	fused := touch("fused", "laya.onnx")
	touch("splitdir", "encoder.onnx")
	touch("splitdir", "head.onnx")
	half := touch("half", "encoder.onnx")
	cases := []struct {
		name, path, problem string
	}{
		{"fused file", fused, ""},
		{"half a split pair is only a file", half, ""},
		{"missing file", filepath.Join(dir, "fused", "nope.onnx"), "no such file"},
		{"directory", filepath.Join(dir, "fused"), "is a directory"},
		{"split directory", filepath.Join(dir, "splitdir"), "split encoder.onnx + head.onnx layout is not supported"},
		{"missing laya.onnx beside a split pair", filepath.Join(dir, "splitdir", "laya.onnx"), "no such file; the split"},
		{"the encoder half itself", filepath.Join(dir, "splitdir", "encoder.onnx"), "split encoder.onnx + head.onnx layout"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := checkPath(c.path)
			if c.problem == "" {
				if err != nil {
					t.Fatalf("checkPath = %v", err)
				}
				return
			}
			var ge *GraphError
			if !errors.As(err, &ge) || ge.Path != c.path || !strings.Contains(ge.Problem, c.problem) {
				t.Fatalf("checkPath = %v, want a *GraphError for %s containing %q", err, c.path, c.problem)
			}
		})
	}
}

func TestCollect(t *testing.T) {
	b := infer.Batch{Rows: 2, Length: 4, Markers: 3}
	logits := outTensor{[]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}}
	act := outTensor{[]int64{2, 2}, []float32{7, 8, 9, 10}}
	out, err := collect(b, logits, act)
	if err != nil || len(out.Logits) != 6 || len(out.ActLogits) != 4 || out.Logits[5] != 6 || out.ActLogits[3] != 10 {
		t.Fatalf("collect = %+v, %v", out, err)
	}
	out.Logits[0] = 99
	if logits.data[0] != 1 {
		t.Error("collect returned the runtime's memory, not a copy")
	}

	cases := []struct {
		name         string
		logits, act  outTensor
		want         string
		shapeErrName string
	}{
		{"transposed logits of the same size", outTensor{[]int64{3, 2}, logits.data}, act, "output logits has shape [3 2], want [2 3]", ""},
		{"act_logits in the logits slot", outTensor{[]int64{2, 2}, act.data}, logits, "output logits has shape [2 2], want [2 3]", ""},
		{"logits in the act_logits slot", logits, outTensor{[]int64{2, 3}, logits.data}, "output act_logits has shape [2 3], want [2 2]", ""},
		{"rank 1", outTensor{[]int64{6}, logits.data}, act, "output logits has shape [6]", ""},
		{"short data under the right shape", outTensor{[]int64{2, 3}, logits.data[:5]}, act, "logits has 5 values, want 6", "logits"},
		{"short act data under the right shape", logits, outTensor{[]int64{2, 2}, act.data[:3]}, "act_logits has 3 values, want 4", "act_logits"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := collect(b, c.logits, c.act)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("collect = %v, want %q", err, c.want)
			}
			var se *infer.ShapeError
			if (c.shapeErrName != "") != errors.As(err, &se) || (se != nil && se.Name != c.shapeErrName) {
				t.Fatalf("collect = %v, want ShapeError for %q", err, c.shapeErrName)
			}
		})
	}
}

func TestLibraryPath(t *testing.T) {
	cases := []struct{ explicit, env, want string }{
		{"/a/lib.so", "/b/lib.so", "/a/lib.so"},
		{"", "/b/lib.so", "/b/lib.so"},
		{"/a/lib.so", "", "/a/lib.so"},
	}
	for _, c := range cases {
		if got := libraryPath(c.explicit, c.env); got != c.want {
			t.Errorf("libraryPath(%q, %q) = %q, want %q", c.explicit, c.env, got, c.want)
		}
	}
	for goos, want := range map[string]string{
		"darwin": "libonnxruntime.dylib", "linux": "libonnxruntime.so", "windows": "onnxruntime.dll", "freebsd": "libonnxruntime.so",
	} {
		if got := defaultLibrary(goos); got != want {
			t.Errorf("defaultLibrary(%s) = %q, want %q", goos, got, want)
		}
	}
	if got := libraryPath("", ""); got != defaultLibrary(runtime.GOOS) {
		t.Errorf("libraryPath(\"\", \"\") = %q", got)
	}
}

func TestLoadErrorHint(t *testing.T) {
	err := loadError("/x/libonnxruntime.so", errors.New("Error setting ORT API base: 2"))
	if !errors.Is(err, ErrRuntime) {
		t.Fatalf("%v does not wrap ErrRuntime", err)
	}
	for _, want := range []string{`"/x/libonnxruntime.so"`, "Error setting ORT API base", "1.29 or newer", LibraryEnv, "WithLibrary"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("%q lacks %q", err, want)
		}
	}
}

func TestLoadErrorHintOnlyForAPIBaseFailures(t *testing.T) {
	err := loadError("/x/missing.so", errors.New("dlopen: no such file or directory"))
	if !errors.Is(err, ErrRuntime) || !strings.Contains(err.Error(), "no such file") {
		t.Fatalf("%v", err)
	}
	if strings.Contains(err.Error(), "1.29") {
		t.Errorf("a file-not-found error carries the version hint: %v", err)
	}
}

func TestResolveOptions(t *testing.T) {
	o, err := resolve([]Option{WithLibrary("/l"), nil, WithIntraOpThreads(3), WithInterOpThreads(2)})
	if err != nil || o.library != "/l" || o.intra != 3 || o.inter != 2 {
		t.Fatalf("resolve = %+v, %v", o, err)
	}
	for _, opt := range []Option{WithIntraOpThreads(-1), WithInterOpThreads(-1)} {
		if _, err := resolve([]Option{opt}); err == nil || !strings.HasPrefix(err.Error(), "onnx: ") {
			t.Errorf("negative threads: %v", err)
		}
	}
}

func TestSamePath(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "lib.so")
	if err := os.WriteFile(file, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.so")
	if err := os.Symlink(file, link); err != nil {
		t.Skip("no symlinks here")
	}
	if !samePath(file, link) || !samePath(file, filepath.Join(dir, ".", "lib.so")) {
		t.Error("one file under two spellings was called different")
	}
	if samePath(file, filepath.Join(dir, "other.so")) || samePath("/nonexistent/a.so", "/nonexistent/b.so") {
		t.Error("different files were called the same")
	}
}
