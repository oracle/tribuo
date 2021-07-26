/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.onnx;

import ai.onnx.proto.OnnxMl;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The supported ONNX operators.
 */
public enum ONNXOperators {


    /**
     * Identity,
     */
    IDENTITY("Identity",1,1),
    /**
     * Sigmoid element-wise.
     */
    SIGMOID("Sigmoid",1,1),
    /**
     * Softmax.
     */
    SOFTMAX("Softmax",1,1, Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT)
    )),
    /**
     * Element-wise addition with broadcasting.
     */
    ADD("Add",2,1),
    /**
     * Element-wise subtraction with broadcasting.
     */
    SUB("Sub",2,1),
    /**
     * Element-wise multiplication with broadcasting.
     */
    MUL("Mul",2,1),
    /**
     * Element-wise division with broadcasting.
     */
    DIV("Div",2,1),
    /**
     * Compute the minimum along the specified axes of the tensor.
     */
    REDUCE_MIN("ReduceMin",1,1,Arrays.asList(
            new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS),
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT)
    )),
    /**
     * Compute the sum along the specified axes of the tensor.
     */
    REDUCE_SUM("ReduceSum",2,1,Arrays.asList(
            new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS), //Opset 11
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT)
    )),
    /**
     * General Matrix Multiply: alpha*AB + beta*C.
     */
    GEMM("Gemm",3,1, Arrays.asList(
            new ONNXAttribute("alpha", OnnxMl.AttributeProto.AttributeType.FLOAT),
            new ONNXAttribute("beta", OnnxMl.AttributeProto.AttributeType.FLOAT),
            new ONNXAttribute("transA", OnnxMl.AttributeProto.AttributeType.INT),
            new ONNXAttribute("transB", OnnxMl.AttributeProto.AttributeType.INT)
    ));

    /**
     * The operator name.
     */
    public final String opName;
    /**
     * The number of inputs.
     */
    public final int numInputs;
    /**
     * The number of outputs.
     */
    public final int numOutputs;
    /**
     * The operator attributes.
     */
    public final Map<String,ONNXAttribute> attributes;

    /**
     * Opset supported by these definitions.
     */
    private static final int OPSET_VERSION = 11;

    private ONNXOperators(String value, int numInputs, int numOutputs) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.attributes = Collections.emptyMap();
    }

    private ONNXOperators(String value, int numInputs, int numOutputs, List<ONNXAttribute> attributes) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        Map<String,ONNXAttribute> attributeMap = new HashMap<>();
        for (ONNXAttribute a : attributes) {
            attributeMap.put(a.getName(),a);
        }
        if (attributes.size() != attributeMap.size()) {
            throw new IllegalArgumentException("Duplicate attribute in enum declaration - " + attributes);
        }
        this.attributes = Collections.unmodifiableMap(attributeMap);
    }

    /**
     * Builds this node based on the supplied inputs and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param outputs The names of the outputs.
     * @return The NodeProto.
     */
    public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String[] outputs) {
        return build(context,inputs,outputs,Collections.emptyMap());
    }

    /**
     * Builds this node based on the supplied inputs and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param outputs The names of the outputs.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String[] outputs, Map<String,Object> attributeValues) {
        if (inputs.length != numInputs) {
            throw new IllegalArgumentException("Expected " + numInputs + " inputs, but received " + inputs.length);
        }
        if (outputs.length != numOutputs) {
            throw new IllegalArgumentException("Expected " + numOutputs + " outputs, but received " + outputs.length);
        }
        if (attributeValues.size() > attributes.size()) {
            throw new IllegalArgumentException("Found more attributes than expected, received " + attributeValues.size() + ", expected at most " + attributes.size());
        }
        if (!attributes.keySet().containsAll(attributeValues.keySet())) {
            throw new IllegalArgumentException("Unexpected attribute found, received " + attributeValues.keySet() + ", expected " + attributes.keySet());
        }
        OnnxMl.NodeProto.Builder nodeBuilder = OnnxMl.NodeProto.newBuilder();
        for (String i : inputs) {
            nodeBuilder.addInput(i);
        }
        for (String o : outputs) {
            nodeBuilder.addOutput(o);
        }
        nodeBuilder.setName(context.generateUniqueName(opName));
        nodeBuilder.setOpType(opName);
        for (Map.Entry<String,Object> e : attributeValues.entrySet()) {
            ONNXAttribute attr = attributes.get(e.getKey());
            nodeBuilder.addAttribute(attr.build(e.getValue()));
        }
        return nodeBuilder.build();
    }

    /**
     * Returns the opset version supported by these operators.
     * @return The opset version.
     */
    public static int getOpsetVersion() {
        return OPSET_VERSION;
    }

    /**
     * Returns the opset proto for these operators.
     * @return The opset proto.
     */
    public static OnnxMl.OperatorSetIdProto getOpsetProto() {
        return OnnxMl.OperatorSetIdProto.newBuilder().setVersion(ONNXOperators.getOpsetVersion()).build();
    }
}
