package org.tribuo.onnx;

import ai.onnx.proto.OnnxMl;

public class ONNXPlaceholder extends ONNXRef<OnnxMl.ValueInfoProto> {

    ONNXPlaceholder(ONNXContext onnx, OnnxMl.ValueInfoProto backRef, String basename) {
        super(onnx, backRef, basename);
    }

    @Override
    public String getReference() {
        return backRef.getName();
    }
}
