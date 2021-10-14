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

package org.tribuo.math.onnx;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.ByteString;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXShape;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Helper functions for building ONNX protos.
 */
public abstract class ONNXMathUtils {

    /**
     * Private constructor for abstract util class.
     */
    private ONNXMathUtils() {}

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

    /**
     * Generic method to create float {@link ai.onnx.proto.OnnxMl.TensorProto} instances.
     *
     * @param context the naming context.
     * @param name the base name for the proto.
     * @param dims the dimensions of the input data.
     * @param dataPopulator a method to populate a {@link FloatBuffer} that will be written into the TensorProto's rawData field.
     * @return a float-typed TensorProto representation of the data.
     */
    public static OnnxMl.TensorProto floatTensorBuilder(ONNXContext context, String name, List<Integer> dims, Consumer<FloatBuffer> dataPopulator) {
        int size = dims.stream().reduce((a, b) -> a * b).orElse(0);
        ByteBuffer buffer = ByteBuffer.allocate(size * 4).order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        dataPopulator.accept(floatBuffer);
        floatBuffer.rewind();
        return OnnxMl.TensorProto.newBuilder()
                .setName(context.generateUniqueName(name))
                .setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber())
                .addAllDims(() -> dims.stream().map(Integer::longValue).iterator())
                .setRawData(ByteString.copyFrom(buffer))
                .build();
    }

    /**
     * Generic method to create double {@link ai.onnx.proto.OnnxMl.TensorProto} instances.
     * <p>
     * Note that ONNX fp64 support is poor compared to fp32.
     * @param context the naming context.
     * @param name the base name for the proto.
     * @param dims the dimensions of the input data.
     * @param dataPopulator a method to populate a {@link DoubleBuffer} that will be written into the TensorProto's rawData field.
     * @return a double-typed TensorProto representation of the data.
     */
    public static OnnxMl.TensorProto doubleTensorBuilder(ONNXContext context, String name, List<Integer> dims, Consumer<DoubleBuffer> dataPopulator) {
        int size = dims.stream().reduce((a, b) -> a * b).orElse(0);
        ByteBuffer buffer = ByteBuffer.allocate(size * 8).order(ByteOrder.LITTLE_ENDIAN);
        DoubleBuffer doubleBuffer = buffer.asDoubleBuffer();
        dataPopulator.accept(doubleBuffer);
        doubleBuffer.rewind();
        return OnnxMl.TensorProto.newBuilder()
                .setName(context.generateUniqueName(name))
                .setDataType(OnnxMl.TensorProto.DataType.DOUBLE.getNumber())
                .addAllDims(() -> dims.stream().map(Integer::longValue).iterator())
                .setRawData(ByteString.copyFrom(buffer))
                .build();
    }

    /**
     * Builds a TensorProto containing the array.
     * <p>
     * Downcasts the doubles into floats as ONNX's fp64 support is poor compared to fp32.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param parameters The array to store in the proto.
     * @return A TensorProto containing the array as floats.
     */
    public static OnnxMl.TensorProto arrayBuilder(ONNXContext context, String name, double[] parameters) {
        return arrayBuilder(context,name,parameters,true);
    }

    /**
     * Builds a TensorProto containing the array.
     * <p>
     * Optionally downcasts the doubles into floats.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param parameters The array to store in the proto.
     * @param downcast Downcasts the doubles into floats.
     * @return A TensorProto containing the array as either floats or doubles.
     */
    public static OnnxMl.TensorProto arrayBuilder(ONNXContext context, String name, double[] parameters, boolean downcast) {
        return downcast
                ? floatTensorBuilder(context, name, Collections.singletonList(parameters.length),
                fb -> Arrays.stream(parameters).forEachOrdered(d -> fb.put((float)d)))
                : doubleTensorBuilder(context, name, Collections.singletonList(parameters.length),
                db -> Arrays.stream(parameters).forEachOrdered(db::put));
    }

    /**
     * Builds a TensorProto containing the {@link SGDVector}.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param vector the SGDVector to store in the proto.
     * @return A TensorProto containing the vector.
     */
    public static OnnxMl.TensorProto floatVectorBuilder(ONNXContext context, String name, SGDVector vector) {
        return floatTensorBuilder(context, name, Collections.singletonList(vector.size()),
                fb -> vector.forEach(vt -> fb.put(vt.index,(float) vt.value)));
    }

    /**
     * Builds a TensorProto containing the {@link Matrix}.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param matrix the matrix to store in the proto.
     * @param transpose Whether to transpose the vector before writing it.
     * @return A TensorProto containing the matrix
     */
    public static OnnxMl.TensorProto floatMatrixBuilder(ONNXContext context, String name, Matrix matrix, boolean transpose) {
        return floatTensorBuilder(context, name,
                Arrays.stream(matrix.getShape()).boxed().collect(Collectors.toList()),
                fb -> matrix.forEach(mt -> {
                    int address = transpose
                            ? mt.j * matrix.getDimension2Size() + mt.i
                            : mt.i * matrix.getDimension1Size() + mt.j;
                    System.out.println("tuple: " + mt.toString() + " address: " +address + " buffersize:" + fb.capacity());
                    fb.put(address, (float) mt.value);
                }));
    }
}
