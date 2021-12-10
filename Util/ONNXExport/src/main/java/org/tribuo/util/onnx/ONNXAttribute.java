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

package org.tribuo.util.onnx;

import com.google.protobuf.ByteString;
import ai.onnx.proto.OnnxMl;

import java.nio.charset.StandardCharsets;

/**
 * The spec for an attribute, used to produce the attribute proto at construction time.
 */
public final class ONNXAttribute {

    /**
     * Value used to denote a varaidic input in an ONNX operator
     */
    public static final int VARIADIC_INPUT = -1;

    private final String name;
    private final OnnxMl.AttributeProto.AttributeType type;
    private final boolean mandatory;

    /**
     * Constructs an attribute placeholder of the appropriate name and type.
     *
     * @param name The name of the attribute.
     * @param type The type of the attribute.
     * @param mandatory Is this attribute mandatory?
     */
    public ONNXAttribute(String name, OnnxMl.AttributeProto.AttributeType type, boolean mandatory) {
        this.name = name;
        this.type = type;
        this.mandatory = mandatory;
    }

    /**
     * Gets the attribute's name.
     *
     * @return The attribute's name.
     */
    public String getName() {
        return name;
    }

    /**
     * Gets the attribute's type.
     *
     * @return The attribute's type.
     */
    public OnnxMl.AttributeProto.AttributeType getType() {
        return type;
    }

    /**
     * Is this attribute mandatory?
     * @return True if the attribute is mandatory for this operation.
     */
    public boolean isMandatory() {
        return mandatory;
    }

    /**
     * Builds the attribute proto using the supplied value.
     * <p>
     * Throws {@link IllegalArgumentException} if the value type does not match the expected type, and
     * throws {@link UnsupportedOperationException} if the value type is not supported. Currently supported
     * types are primitives, strings and arrays of primitives and strings. ONNX attributes only support
     * float and int as primitives.
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
                    builder.setF((Float) value);
                } else {
                    throw new IllegalArgumentException("Expected Float, found " + value.getClass() + " with value " + value);
                }
                break;
            case INT:
                if (value instanceof Integer) {
                    builder.setI((Integer) value);
                } else {
                    throw new IllegalArgumentException("Expected Integer, found " + value.getClass() + " with value " + value);
                }
                break;
            case STRING:
                if (value instanceof String) {
                    builder.setS(ByteString.copyFrom((String) value, StandardCharsets.UTF_8));
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
            case TENSOR:
                if (value instanceof OnnxMl.TensorProto) {
                    builder.setT((OnnxMl.TensorProto) value);
                } else {
                    throw new IllegalArgumentException("Expected TensorProto, found " + value.getClass() + " with value " + value);
                }
                break;
            case GRAPH:
            case SPARSE_TENSOR:
            case TENSORS:
            case GRAPHS:
            case SPARSE_TENSORS:
            case UNDEFINED:
                throw new UnsupportedOperationException("Type: " + type + " is not supported.");
        }

        return builder.build();
    }

    @Override
    public String toString() {
        return "ONNXAttribute(name='" + name + "',type=" + type + ",mandatory="+mandatory+")";
    }
}
