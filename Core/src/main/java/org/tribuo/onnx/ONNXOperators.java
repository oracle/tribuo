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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.tribuo.onnx.ONNXAttribute.VARIADIC_INPUT;

/**
 * The supported ONNX operators.
 */
public enum ONNXOperators {
    /**
     * Identity.
     */
    IDENTITY("Identity",1,1),
    /**
     * Concatenates tensors.
     */
    CONCAT("Concat",VARIADIC_INPUT,1, Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT, true)
    )),
    /**
     * Sigmoid element-wise.
     */
    SIGMOID("Sigmoid",1,1),
    /**
     * Softmax.
     * <ul>
     *     <li>{@code axis} defaults to -1.</li>
     * </ul>
     */
    SOFTMAX("Softmax",1,1, Collections.singletonList(
            new ONNXAttribute("axis", OnnxMl.AttributeProto.AttributeType.INT, false)
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
     * Element-wise exponentiation with broadcasting.
     */
    POW("Pow",2,1),
    /**
     * Compute the minimum along the specified axes of the tensor.
     * <ul>
     *     <li>{@code axes} defaults to all dimensions.</li>
     *     <li>{@code keepdims} defaults to 1 which means keep.</li>
     * </ul>
     */
    REDUCE_MIN("ReduceMin",1,1,Arrays.asList(
            new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS,false),
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT,false)
    )),
    /**
     * Compute the sum along the specified axes of the tensor.
     * <ul>
     *     <li>{@code axes} defaults to all dimensions.</li>
     *     <li>{@code keepdims} defaults to 1 which means keep.</li>
     * </ul>
     */
    REDUCE_SUM("ReduceSum",1,1,Arrays.asList(
            new ONNXAttribute("axes", OnnxMl.AttributeProto.AttributeType.INTS, false), //Opset 11
            new ONNXAttribute("keepdims", OnnxMl.AttributeProto.AttributeType.INT, false)
    )),
    /**
     * General Matrix Multiply: {@code alpha*AB + beta*C}.
     * <p>
     * The {@code C} input is optional, and if not supplied is treated as zero.
     * <ul>
     *     <li>{@code alpha} defaults to 1.0</li>
     *     <li>{@code beta} defaults to 1.0</li>
     *     <li>{@code transA} defaults to 0 (i.e., not transposed)</li>
     *     <li>{@code transB} defaults to 0 (i.e., not transposed)</li>
     * </ul>
     */
    GEMM("Gemm",2,1, 1, Arrays.asList(
            new ONNXAttribute("alpha", OnnxMl.AttributeProto.AttributeType.FLOAT,false),
            new ONNXAttribute("beta", OnnxMl.AttributeProto.AttributeType.FLOAT,false),
            new ONNXAttribute("transA", OnnxMl.AttributeProto.AttributeType.INT,false),
            new ONNXAttribute("transB", OnnxMl.AttributeProto.AttributeType.INT,false)
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
     * The number of optional inputs.
     */
    public final int numOptionalInputs;
    /**
     * The number of outputs.
     */
    public final int numOutputs;
    /**
     * The operator attributes.
     */
    public final Map<String,ONNXAttribute> attributes;
    /**
     * The mandatory attribute names.
     */
    public final Set<String> mandatoryAttributeNames;

    /**
     * Opset supported by these definitions.
     */
    private static final int OPSET_VERSION = 11;

    /**
     * Builds an operator without attributes.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOutputs The number of outputs.
     */
    private ONNXOperators(String value, int numInputs, int numOutputs) {
        this(value,numInputs,0,numOutputs);
    }

    /**
     * Builds an operator without attributes and with optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOptionalInputs The number of optional inputs.
     * @param numOutputs The number of outputs.
     */
    private ONNXOperators(String value, int numInputs, int numOptionalInputs, int numOutputs) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOptionalInputs = numOptionalInputs;
        this.numOutputs = numOutputs;
        this.attributes = Collections.emptyMap();
        this.mandatoryAttributeNames = Collections.emptySet();
    }

    /**
     * Builds an operator with attributes.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOutputs The number of outputs.
     * @param attributes The attributes.
     */
    private ONNXOperators(String value, int numInputs, int numOutputs, List<ONNXAttribute> attributes) {
        this(value,numInputs,0,numOutputs,attributes);
    }

    /**
     * Builds an operator with attributes and optional inputs.
     * @param value The operator name.
     * @param numInputs The number of inputs.
     * @param numOptionalInputs The number of optional inputs.
     * @param numOutputs The number of outputs.
     * @param attributes The attributes.
     */
    private ONNXOperators(String value, int numInputs, int numOptionalInputs, int numOutputs, List<ONNXAttribute> attributes) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOptionalInputs = numOptionalInputs;
        this.numOutputs = numOutputs;
        Map<String,ONNXAttribute> attributeMap = new HashMap<>();
        Set<String> attributeSet = new HashSet<>();
        for (ONNXAttribute a : attributes) {
            attributeMap.put(a.getName(),a);
            if (a.isMandatory()) {
                attributeSet.add(a.getName());
            }
        }
        if (attributes.size() != attributeMap.size()) {
            throw new IllegalArgumentException("Duplicate attribute in enum declaration - " + attributes);
        }
        this.attributes = Collections.unmodifiableMap(attributeMap);
        this.mandatoryAttributeNames = attributeSet.isEmpty() ? Collections.emptySet() : Collections.unmodifiableSet(attributeSet);
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if this operator takes more than a single input or output.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param output The name of the output.
     * @return The NodeProto.
     */
    public OnnxMl.NodeProto build(ONNXContext context, String input, String output) {
        return build(context,new String[]{input},new String[]{output},Collections.emptyMap());
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if this operator takes more than a single input or output.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The names of the input.
     * @param output The name of the output.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    public OnnxMl.NodeProto build(ONNXContext context, String input, String output, Map<String,Object> attributeValues) {
        return build(context,new String[]{input},new String[]{output},attributeValues);
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param output The name of the output.
     * @return The NodeProto.
     */
    public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String output) {
        return build(context,inputs,new String[]{output},Collections.emptyMap());
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param output The name of the output.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String output, Map<String,Object> attributeValues) {
        return build(context,inputs,new String[]{output},attributeValues);
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
        if ((numInputs != VARIADIC_INPUT) && ((inputs.length < numInputs) || (inputs.length > numInputs + numOptionalInputs))) {
            throw new IllegalArgumentException("Expected " + numInputs + " inputs, with " + numOptionalInputs + " optional inputs, but received " + inputs.length);
        }
        if (outputs.length != numOutputs) {
            throw new IllegalArgumentException("Expected " + numOutputs + " outputs, but received " + outputs.length);
        }
        if (attributeValues.size() > attributes.size()) {
            throw new IllegalArgumentException("Found more attributes than expected, received " + attributeValues.size() + ", expected at most " + attributes.size());
        }
        if (!attributes.keySet().containsAll(attributeValues.keySet())) {
            throw new IllegalArgumentException("Unexpected attribute found, received " + attributeValues.keySet() + ", expected values from " + attributes.keySet());
        }
        if (!attributeValues.keySet().containsAll(mandatoryAttributeNames)) {
            throw new IllegalArgumentException("Expected to find all mandatory attributes, received " + attributeValues.keySet() + ", expected " + mandatoryAttributeNames);
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
