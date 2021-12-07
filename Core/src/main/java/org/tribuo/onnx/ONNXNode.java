package org.tribuo.onnx;

import ai.onnx.proto.OnnxMl;

public class ONNXNode extends ONNXRef<OnnxMl.NodeProto> {
    //protected OnnxMl.NodeProto backRef;
    private final int outputIndex;

    ONNXNode(ONNXContext onnx, OnnxMl.NodeProto backRef, String basename, int outputIndex) {
        super(onnx, backRef, basename);
        this.outputIndex = outputIndex;
    }

    @Override
    public String getReference() {
        return backRef.getOutput(outputIndex);
    }
}
