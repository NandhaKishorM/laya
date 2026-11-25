module github.com/NandhaKishorM/laya/laya-go/samples

go 1.22

require (
	github.com/NandhaKishorM/laya/laya-go v0.0.0-00010101000000-000000000000
	github.com/NandhaKishorM/laya/laya-go/onnx v0.0.0-00010101000000-000000000000
)

require github.com/yalue/onnxruntime_go v1.36.0 // indirect

replace (
	github.com/NandhaKishorM/laya/laya-go => ../
	github.com/NandhaKishorM/laya/laya-go/onnx => ../onnx
)
