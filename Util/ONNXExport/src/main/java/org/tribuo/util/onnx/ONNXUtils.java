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

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.ByteString;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

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

    /**
     * Builds a TensorProto containing the scalar value.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param value The value to store.
     * @return A TensorProto containing the value as an int.
     */
    public static OnnxMl.TensorProto scalarBuilder(ONNXContext context, String name, int value) {
        OnnxMl.TensorProto.Builder scalarBuilder = OnnxMl.TensorProto.newBuilder();
        scalarBuilder.setName(context.generateUniqueName(name));
        scalarBuilder.setDataType(OnnxMl.TensorProto.DataType.INT32.getNumber());
        scalarBuilder.addInt32Data(value);
        return scalarBuilder.build();
    }

    /**
     * Builds a TensorProto containing the scalar value.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param value The value to store.
     * @return A TensorProto containing the value as a long.
     */
    public static OnnxMl.TensorProto scalarBuilder(ONNXContext context, String name, long value) {
        OnnxMl.TensorProto.Builder scalarBuilder = OnnxMl.TensorProto.newBuilder();
        scalarBuilder.setName(context.generateUniqueName(name));
        scalarBuilder.setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber());
        scalarBuilder.addInt64Data(value);
        return scalarBuilder.build();
    }

    /**
     * Builds a TensorProto containing the scalar value.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param value The value to store.
     * @return A TensorProto containing the value as a float.
     */
    public static OnnxMl.TensorProto scalarBuilder(ONNXContext context, String name, float value) {
        OnnxMl.TensorProto.Builder scalarBuilder = OnnxMl.TensorProto.newBuilder();
        scalarBuilder.setName(context.generateUniqueName(name));
        scalarBuilder.setDataType(OnnxMl.TensorProto.DataType.FLOAT.getNumber());
        scalarBuilder.addFloatData(value);
        return scalarBuilder.build();
    }

    /**
     * Builds a TensorProto containing the scalar value.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param value The value to store.
     * @return A TensorProto containing the value as a double.
     */
    public static OnnxMl.TensorProto scalarBuilder(ONNXContext context, String name, double value) {
        OnnxMl.TensorProto.Builder scalarBuilder = OnnxMl.TensorProto.newBuilder();
        scalarBuilder.setName(context.generateUniqueName(name));
        scalarBuilder.setDataType(OnnxMl.TensorProto.DataType.DOUBLE.getNumber());
        scalarBuilder.addDoubleData(value);
        return scalarBuilder.build();
    }

    /**
     * Generic method to create float {@link OnnxMl.TensorProto} instances.
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
                .addAllDims(dims.stream().map(Integer::longValue).collect(Collectors.toList()))
                .setRawData(ByteString.copyFrom(buffer))
                .build();
    }

    /**
     * Generic method to create double {@link OnnxMl.TensorProto} instances.
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
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param parameters The array to store in the proto.
     * @return A TensorProto containing the array as floats.
     */
    public static OnnxMl.TensorProto arrayBuilder(ONNXContext context, String name, float[] parameters) {
        return floatTensorBuilder(context, name, Collections.singletonList(parameters.length),
                (FloatBuffer fb) -> fb.put(parameters));
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
        if(downcast) {
            return floatTensorBuilder(context, name, Collections.singletonList(parameters.length),
                    (FloatBuffer fb) -> Arrays.stream(parameters).forEachOrdered(d -> fb.put((float)d)));
        } else {
            return doubleTensorBuilder(context, name, Collections.singletonList(parameters.length),
                    (DoubleBuffer db) -> Arrays.stream(parameters).forEachOrdered(db::put));
        }
    }

    /**
     * Builds a TensorProto containing the array.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param parameters The array to store in the proto.
     * @return A TensorProto containing the array as ints.
     */
    public static OnnxMl.TensorProto arrayBuilder(ONNXContext context, String name, int[] parameters) {
        OnnxMl.TensorProto.Builder arrBuilder = OnnxMl.TensorProto.newBuilder();
        arrBuilder.setName(context.generateUniqueName(name));
        arrBuilder.addDims(parameters.length);
        int capacity = parameters.length * 4;
        ByteBuffer buffer = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
        arrBuilder.setDataType(OnnxMl.TensorProto.DataType.INT32.getNumber());
        IntBuffer intBuffer = buffer.asIntBuffer();
        intBuffer.put(parameters);
        intBuffer.rewind();
        arrBuilder.setRawData(ByteString.copyFrom(buffer));
        return arrBuilder.build();
    }

    /**
     * Builds a TensorProto containing the array.
     * @param context The naming context.
     * @param name The base name for the proto.
     * @param parameters The array to store in the proto.
     * @return A TensorProto containing the array as longs.
     */
    public static OnnxMl.TensorProto arrayBuilder(ONNXContext context, String name, long[] parameters) {
        OnnxMl.TensorProto.Builder arrBuilder = OnnxMl.TensorProto.newBuilder();
        arrBuilder.setName(context.generateUniqueName(name));
        arrBuilder.addDims(parameters.length);
        int capacity = parameters.length * 8;
        ByteBuffer buffer = ByteBuffer.allocate(capacity).order(ByteOrder.LITTLE_ENDIAN);
        arrBuilder.setDataType(OnnxMl.TensorProto.DataType.INT64.getNumber());
        LongBuffer longBuffer = buffer.asLongBuffer();
        longBuffer.put(parameters);
        longBuffer.rewind();
        arrBuilder.setRawData(ByteString.copyFrom(buffer));
        return arrBuilder.build();
    }
}
