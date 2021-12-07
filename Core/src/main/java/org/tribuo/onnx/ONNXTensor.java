package org.tribuo.onnx;

import ai.onnx.proto.OnnxMl;

public class ONNXTensor extends ONNXRef<OnnxMl.TensorProto> {
    ONNXTensor(ONNXContext onnx, OnnxMl.TensorProto backRef, String baseName) {
        super(onnx, backRef, baseName);
    }

    @Override
    public String getReference() {
        return backRef.getName();
    }
}
