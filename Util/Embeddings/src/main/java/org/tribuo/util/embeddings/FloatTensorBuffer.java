/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.embeddings;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A tensor containing primitive floats in a buffer.
 */
public final class FloatTensorBuffer extends TensorBuffer<FloatBuffer> {

    /**
     * Creates a float tensor from the supplied buffer and shape.
     * @param buffer The buffer.
     * @param shape The shape.
     */
    public FloatTensorBuffer(FloatBuffer buffer, long[] shape) {
        super(buffer, shape);
    }

    /**
     * Creates a float tensor from the supplied buffer and shape filled with the supplied value.
     * @param buffer The buffer.
     * @param shape The shape.
     * @param value The value.
     */
    public FloatTensorBuffer(FloatBuffer buffer, long[] shape, float value) {
        super(buffer, shape);

        for (int i = 0; i < this.numElements; i++) {
            buffer.put(value);
        }
        buffer.rewind();
    }

    /**
     * Creates an empty float tensor of the supplied shape backed by a direct byte buffer.
     * @param shape The shape.
     */
    public FloatTensorBuffer(long[] shape) {
        this(shape, true);
    }

    /**
     * Creates an empty float tensor of the supplied shape backed by a byte buffer.
     * @param shape The shape.
     */
    public FloatTensorBuffer(long[] shape, boolean direct) {
        super(alloc(shape, direct), shape);
    }

    @Override
    public FloatTensorBuffer copy() {
        FloatBuffer copy = alloc(shape, buffer.isDirect());
        copy.put(buffer);
        copy.rewind();
        buffer.rewind();
        return new FloatTensorBuffer(copy, Arrays.copyOf(shape, shape.length));
    }

    /**
     * Splits this tensor into a list of new {@code FloatTensorBuffer}s.
     * <p>
     * The tensors are split in linear row major order, partitioned on the leading dimension.
     * @throws IllegalArgumentException If the supplied shape does not split this tensor in equal chunks.
     * @param newShape The new shape for the tensors.
     * @return A list containing the new tensors.
     */
    public List<FloatTensorBuffer> split(long[] newShape) {
        int newNumElements = computeNumElements(newShape);
        if (numElements % newNumElements != 0) {
            throw new IllegalArgumentException("Invalid shape for splitting, expected to split in into equal chunks.");
        }
        int numTensors = numElements / newNumElements;
        List<FloatTensorBuffer> output = new ArrayList<>();
        int position = 0;
        for (int i = 0; i < numTensors; i++) {
            FloatTensorBuffer tensor = new FloatTensorBuffer(newShape);
            tensor.buffer.put(0, buffer, position, tensor.numElements);
            position += tensor.numElements;
            output.add(tensor);
        }
        return output;
    }

    @Override
    public OnnxTensor wrapForORT(OrtEnvironment env) throws OrtException {
        return OnnxTensor.createTensor(env, this.buffer, this.shape);
    }

    /**
     * Normalizes this tensor so the last dimension forms a unit vector.
     */
    public void l2InPlace() {
        int rowLength = (int) shape[shape.length-1];
        int numRows = 1;
        for (int i = 0; i < shape.length - 1; i++) {
            numRows *= (int) shape[i];
        }
        for (int i = 0; i < numRows; i++) {
            int offset = i * rowLength;
            float sum = 0.0f;
            for (int j = 0; j < rowLength; j++) {
                float tmp = buffer.get(j + offset);
                sum += tmp * tmp;
            }
            sum = (float) Math.sqrt(sum);
            for (int j = 0; j < rowLength; j++) {
                int idx = j + offset;
                buffer.put(idx, buffer.get(idx) / sum);
            }
        }
    }

    /**
     * Adds the supplied tensor to this one.
     * @throws IllegalArgumentException If the other tensor is not the same shape as this one.
     * @param t The tensor to add.
     */
    public void add(FloatTensorBuffer t) {
        if (!Arrays.equals(t.shape,shape)) {
            throw new IllegalArgumentException("Invalid shape. Expected " + Arrays.toString(shape) + ", found " + Arrays.toString(t.shape));
        }
        for (int i = 0; i < numElements; i++) {
            buffer.put(i, buffer.get(i) + t.buffer.get(i));
        }
    }

    /**
     * Scales each element of the buffer by the supplied float.
     * <p>
     * Leaves the buffer position unchanged.
     * @param scalar The scalar.
     */
    public void scale(float scalar) {
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i,buffer.get(i)*scalar);
        }
    }

    /**
     * Gets an element from this tensor.
     * @param idxArr The index to return.
     * @return The element at the index.
     */
    public float get(long... idxArr) {
        int idx = computeIdx(idxArr);
        return buffer.get(idx);
    }

    /**
     * Returns a flat array copy of this tensor.
     * @return The values in the tensor.
     */
    public float[] getFlatArray() {
        float[] output = new float[buffer.capacity()];
        buffer.get(output);
        return output;
    }

    /**
     * Creates a direct {@link FloatBuffer} with capacity equal to the supplied shape.
     * @throws IllegalArgumentException if the shape is larger than the largest buffer.
     * @param shape The shape.
     * @return An int buffer.
     */
    private static FloatBuffer alloc(long[] shape, boolean direct) {
        int elements = computeNumElements(shape);
        if (elements < 0) {
            throw new IllegalArgumentException("Invalid shape for Java tensor, expected less than Integer.MAX_VALUE elements, found " + Arrays.toString(shape));
        }
        if (direct) {
            return ByteBuffer.allocateDirect(elements * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();
        } else {
            return FloatBuffer.allocate(elements);
        }
    }
}
