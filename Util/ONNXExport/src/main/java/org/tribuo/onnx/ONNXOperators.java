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

import com.google.protobuf.ByteString;
import onnx.OnnxMl;

import java.nio.charset.StandardCharsets;
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
     * Softmax
     */
    SOFTMAX("Softmax",1,1,Arrays.asList(
            new ONNXAttribute("axis",OnnxMl.AttributeProto.AttributeType.INT)
    )),
    /**
     * Generalized Matrix Multiply
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

    private ONNXOperators(String value, int numInputs, int numOutputs, List<ONNXAttribute> attributes) {
        this.opName = value;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        Map<String,ONNXAttribute> attributeMap = new HashMap<>();
        for (ONNXAttribute a : attributes) {
            attributeMap.put(a.name,a);
        }
        if (attributes.size() != attributeMap.size()) {
            throw new IllegalArgumentException("Duplicate attribute in enum declaration - " + attributes);
        }
        this.attributes = Collections.unmodifiableMap(attributeMap);
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
            throw new IllegalArgumentException("Found more attributes than expected, received " + attributeValues.size() + ", expected " + attributes.size());
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
     * The spec for an attribute, used to produce the attribute proto at construction time.
     */
    public static final class ONNXAttribute {
        private final String name;
        private final OnnxMl.AttributeProto.AttributeType type;

        /**
         * Constructs an attribute placeholder of the appropriate name and type.
         * @param name The name of the attribute.
         * @param type The type of the attribute.
         */
        public ONNXAttribute(String name, OnnxMl.AttributeProto.AttributeType type) {
            this.name = name;
            this.type = type;
        }

        /**
         * Gets the attribute's name.
         * @return The attribute's name.
         */
        public String getName() {
            return name;
        }

        /**
         * Builds the attribute proto using the supplied value.
         * <p>
         * Throws {@link IllegalArgumentException} if the value type does not match the expected type, and
         * throws {@link UnsupportedOperationException} if the value type is not supported. Currently supported
         * types are primitives, strings and arrays of primitives and strings.
         * @param value The value
         * @return The AttributeProto.
         */
        public OnnxMl.AttributeProto build(Object value) {
            OnnxMl.AttributeProto.Builder builder = OnnxMl.AttributeProto.newBuilder();

            builder.setName(name);
            builder.setType(type);
            switch (type) {
                case FLOAT:
                    if (value instanceof Float) {
                        builder.setF((Float)value);
                    } else {
                        throw new IllegalArgumentException("Expected Float, found " + value.getClass() + " with value " + value);
                    }
                    break;
                case INT:
                    if (value instanceof Integer) {
                        builder.setI((Integer)value);
                    } else {
                        throw new IllegalArgumentException("Expected Integer, found " + value.getClass() + " with value " + value);
                    }
                    break;
                case STRING:
                    if (value instanceof String) {
                        builder.setS(ByteString.copyFrom((String)value, StandardCharsets.UTF_8));
                    } else {
                        throw new IllegalArgumentException("Expected String, found " + value.getClass() + " with value " + value);
                    }
                    break;
                case FLOATS:
                    if (value instanceof float[]) {
                        float[] arr = (float[]) value;
                        for (int i = 0; i < arr.length; i++) {
                            builder.addFloats(arr[i]);
                        }
                    } else {
                        throw new IllegalArgumentException("Expected float[], found " + value.getClass() + " with value " + value);
                    }
                    break;
                case INTS:
                    if (value instanceof int[]) {
                        int[] arr = (int[]) value;
                        for (int i = 0; i < arr.length; i++) {
                            builder.addInts(arr[i]);
                        }
                    } else {
                        throw new IllegalArgumentException("Expected int[], found " + value.getClass() + " with value " + value);
                    }
                    break;
                case STRINGS:
                    if (value instanceof String[]) {
                        String[] arr = (String[]) value;
                        for (int i = 0; i < arr.length; i++) {
                            builder.addStrings(ByteString.copyFrom(arr[i], StandardCharsets.UTF_8));
                        }
                    } else {
                        throw new IllegalArgumentException("Expected String[], found " + value.getClass() + " with value " + value);
                    }
                    break;
                case UNDEFINED:
                case TENSOR:
                case GRAPH:
                case SPARSE_TENSOR:
                case TYPE_PROTO:
                case TENSORS:
                case GRAPHS:
                case SPARSE_TENSORS:
                case TYPE_PROTOS:
                    throw new UnsupportedOperationException("Type: " + type + " is not supported.");
            }

            return builder.build();
        }

        @Override
        public String toString() {
            return "ONNXAttribute(name='"+name+"',type="+type+")";
        }
    }

}
