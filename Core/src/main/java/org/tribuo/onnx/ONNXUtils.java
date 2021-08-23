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

/**
 * Helper functions for building ONNX protos.
 */
public abstract class ONNXUtils {

    /**
     * Private constructor for abstract util class.
     */
    private ONNXUtils() {}

    /**
     * Builds a type proto for the specified shape and tensor type.
     * @param shape The shape.
     * @param type The tensor type.
     * @return The type proto.
     */
    public static OnnxMl.TypeProto buildTensorTypeNode(ONNXShape shape, OnnxMl.TensorProto.DataType type) {
        OnnxMl.TypeProto.Builder builder = OnnxMl.TypeProto.newBuilder();

        OnnxMl.TypeProto.Tensor.Builder tensorBuilder = OnnxMl.TypeProto.Tensor.newBuilder();
        tensorBuilder.setElemType(type.getNumber());
        tensorBuilder.setShape(shape.getProto());
        builder.setTensorType(tensorBuilder.build());

        return builder.build();
    }
}
