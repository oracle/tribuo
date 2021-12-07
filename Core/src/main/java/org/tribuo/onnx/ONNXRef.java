package org.tribuo.onnx;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.GeneratedMessageV3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public abstract class ONNXRef<T extends GeneratedMessageV3> {
    // Unfortunately there is no other shared supertype for OnnxML protobufs
    protected T backRef;
    private final String baseName;
    protected final ONNXContext onnx;


    ONNXRef(ONNXContext onnx, T backRef, String baseName) {
        this.onnx = onnx;
        this.backRef = backRef;
        this.baseName = baseName;
    }

    public abstract String getReference();

    public String getBaseName() {
        return baseName;
    }

    public ONNXContext onnx() {
        return onnx;
    }


    public List<ONNXNode> apply(ONNXOperators op, List<ONNXRef<?>> otherInputs, List<String> outputs, Map<String, Object> attributes) {
        List<ONNXRef<?>> allInputs = new ArrayList<>();
        allInputs.add(this);
        allInputs.addAll(otherInputs);
        return onnx.operation(op, allInputs, outputs, attributes);
    }

    public List<ONNXNode> apply(ONNXOperators op, List<String> outputs, Map<String, Object> attributes) {
        return onnx.operation(op, Collections.singletonList(this), outputs, attributes);
    }

    public ONNXNode apply(ONNXOperators op) {
        return onnx.operation(op, Collections.singletonList(this), getBaseName() + "_" + op.opName, Collections.emptyMap());
    }

    public ONNXNode apply(ONNXOperators op, Map<String, Object> attributes) {
        return onnx.operation(op, Collections.singletonList(this), getBaseName() + "_" + op.opName, attributes);
    }

    public ONNXNode apply(ONNXOperators op, ONNXRef<?> other, Map<String, Object> attributes) {
        return onnx.operation(op, Arrays.asList(this, other), getBaseName() + "_" + op.opName + "_" + other.getBaseName(), attributes);
    }

    public ONNXNode apply(ONNXOperators op, ONNXRef<?> other) {
        return onnx.operation(op, Arrays.asList(this, other), getBaseName() + "_" + op.opName + "_" + other.getBaseName(), Collections.emptyMap());
    }

    public ONNXNode apply(ONNXOperators op, List<ONNXRef<?>> others) {
        return apply(op, others, Collections.singletonList(getBaseName() + "_" + others.stream().map(ONNXRef::getBaseName).collect(Collectors.joining("_"))), Collections.emptyMap()).get(0);
    }

    public ONNXNode apply(ONNXOperators op, List<ONNXRef<?>> others, String outputName) {
        return apply(op, others, Collections.singletonList(outputName), Collections.emptyMap()).get(0);
    }

    public <Ret extends ONNXRef<?>> Ret assignTo(Ret output) {
        return onnx.assignTo(this, output);
    }

    public ONNXNode cast(Class<?> clazz) {
        if (clazz.isAssignableFrom(float.class)) {
            return this.apply(ONNXOperators.CAST, Collections.singletonMap("to", OnnxMl.TensorProto.DataType.FLOAT.getNumber()));
        } else {
            throw new IllegalArgumentException("unsupported class for casting: " + clazz.getName());
        }
    }
}
