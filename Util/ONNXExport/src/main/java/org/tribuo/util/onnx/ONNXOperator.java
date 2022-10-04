/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.onnx;

import ai.onnx.proto.OnnxMl;

import java.util.Collections;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import static org.tribuo.util.onnx.ONNXAttribute.VARIADIC_INPUT;

/**
 * An interface for ONNX operators. Usually implemented by an enum representing the opset.
 */
public interface ONNXOperator {

    /**
     * The operator name.
     * @return The name.
     */
    public String getOpName();

    /**
     * The number of inputs.
     * @return The number of inputs.
     */
    public int getNumInputs();

    /**
     * The number of optional inputs.
     * @return The number of optional inputs.
     */
    public int getNumOptionalInputs();

    /**
     * The number of outputs.
     * @return The number of outputs.
     */
    public int getNumOutputs();

    /**
     * The operator attributes.
     * @return The operator attribute map.
     */
    public Map<String,ONNXAttribute> getAttributes();
    
    /**
     * The mandatory attribute names.
     * @return The required attribute names.
     */
    public Set<String> getMandatoryAttributeNames();

    /**
     * Returns the opset version.
     * @return The opset version.
     */
    public int getOpVersion();

    /**
     * Returns the opset domain.
     * <p>
     * May be {@code null} if it is the default ONNX domain;
     * @return The opset domain.
     */
    public String getOpDomain();

    /**
     * Returns the opset proto for these operators.
     * @return The opset proto.
     */
    default public OnnxMl.OperatorSetIdProto opsetProto() {
        return OnnxMl.OperatorSetIdProto.newBuilder().setDomain(getOpDomain()).setVersion(getOpVersion()).build();
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if this operator takes more than a single input or output.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param output The name of the output.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String output) {
        return build(context,new String[]{input},new String[]{output}, Collections.emptyMap());
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
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String output, Map<String,Object> attributeValues) {
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
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String output) {
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
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String output, Map<String,Object> attributeValues) {
        return build(context,inputs,new String[]{output},attributeValues);
    }

    /**
     * Builds this node based on the supplied input and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param outputs The names of the outputs.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String[] outputs) {
        return build(context,new String[]{input},outputs,Collections.emptyMap());
    }

    /**
     * Builds this node based on the supplied input and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param outputs The names of the outputs.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String[] outputs, Map<String,Object> attributeValues) {
        return build(context,new String[]{input},outputs,attributeValues);
    }

    /**
     * Builds this node based on the supplied inputs and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param outputs The names of the outputs.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String[] outputs) {
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
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String[] outputs, Map<String,Object> attributeValues) {
        int numInputs = getNumInputs();
        int numOptionalInputs = getNumOptionalInputs();
        int numOutputs = getNumOutputs();
        String opName = getOpName();
        String domain = getOpDomain();
        Map<String, ONNXAttribute> attributes = getAttributes();
        Set<String> mandatoryAttributeNames = getMandatoryAttributeNames();

        String opStatus = String.format("Building op %s:%s(%d(+%d)) -> %d", domain, opName, numInputs, numOptionalInputs, numOutputs);

        if ((numInputs != VARIADIC_INPUT) && ((inputs.length < numInputs) || (inputs.length > numInputs + numOptionalInputs))) {
            throw new IllegalArgumentException(opStatus + ". Expected " + numInputs + " inputs, with " + numOptionalInputs + " optional inputs, but received " + inputs.length);
        } else if ((numInputs == VARIADIC_INPUT) && (inputs.length == 0)) {
            throw new IllegalArgumentException(opStatus + ". Expected at least one input for variadic input, received zero");
        }
        if (outputs.length != numOutputs) {
            throw new IllegalArgumentException(opStatus + ". Expected " + numOutputs + " outputs, but received " + outputs.length);
        }
        if (!attributes.keySet().containsAll(attributeValues.keySet())) {
            throw new IllegalArgumentException(opStatus + ". Unexpected attribute found, received " + attributeValues.keySet() + ", expected values from " + attributes.keySet());
        }
        if (!attributeValues.keySet().containsAll(mandatoryAttributeNames)) {
            throw new IllegalArgumentException(opStatus + ". Expected to find all mandatory attributes, received " + attributeValues.keySet() + ", expected " + mandatoryAttributeNames);
        }

        Logger.getLogger("org.tribuo.util.onnx.ONNXOperator").fine(opStatus);
        OnnxMl.NodeProto.Builder nodeBuilder = OnnxMl.NodeProto.newBuilder();
        for (String i : inputs) {
            nodeBuilder.addInput(i);
        }
        for (String o : outputs) {
            nodeBuilder.addOutput(o);
        }
        nodeBuilder.setName(context.generateUniqueName(opName));
        nodeBuilder.setOpType(opName);
        if (domain != null) {
            nodeBuilder.setDomain(domain);
        }
        for (Map.Entry<String,Object> e : attributeValues.entrySet()) {
            ONNXAttribute attr = attributes.get(e.getKey());
            nodeBuilder.addAttribute(attr.build(e.getValue()));
        }
        return nodeBuilder.build();
    }
}
