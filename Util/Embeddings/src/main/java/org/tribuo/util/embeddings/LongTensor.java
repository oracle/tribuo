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
import java.nio.LongBuffer;
import java.util.Arrays;

/**
 * A tensor containing primitive ints in a buffer.
 */
public final class LongTensor extends Tensor<LongBuffer> {

    /**
     * Creates an int tensor from the supplied buffer and shape.
     * @param buffer The buffer.
     * @param shape The shape.
     */
    public LongTensor(LongBuffer buffer, long[] shape) {
        super(buffer, shape);
    }

    /**
     * Creates an empty int tensor of the supplied shape backed by a direct byte buffer.
     * @param shape The shape.
     */
    public LongTensor(long[] shape) {
        super(alloc(shape), shape);
    }

    /**
     * Creates a long tensor of the supplied shape filled with the supplied value.
     * @param shape The shape.
     * @param value The value.
     */
    public LongTensor(long[] shape, long value) {
        super(alloc(shape), shape);

        for (int i = 0; i < this.numElements; i++) {
            buffer.put(value);
        }
        buffer.rewind();
    }

    @Override
    public LongTensor copy() {
        LongBuffer copy = alloc(shape);
        copy.put(buffer);
        copy.rewind();
        buffer.rewind();
        return new LongTensor(copy, Arrays.copyOf(shape, shape.length));
    }

    @Override
    public OnnxTensor wrapForORT(OrtEnvironment env) throws OrtException {
        return OnnxTensor.createTensor(env, this.buffer, this.shape);
    }

    /**
     * Scales each element of the buffer by the supplied integer.
     * <p>
     * Leaves the buffer position unchanged.
     * @param scalar The scalar.
     */
    public void scale(int scalar) {
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i,buffer.get(i)*scalar);
        }
    }

    /**
     * Gets an element from this tensor.
     * @param idxArr The index to return.
     * @return The element at the index.
     */
    public long get(long... idxArr) {
        int idx = computeIdx(idxArr);

        return buffer.get(idx);
    }

    /**
     * Gets the internal buffer representing this Tensor. Use caution when
     * manipulating this buffer as any changes to it will directly modify
     * the internal state of this class.
     *
     * @return the underlying buffer storing the data in this tensor
     */
    public LongBuffer getBuffer() {
        return buffer;
    }

    /**
     * Creates a direct{@link LongBuffer} with capacity equal to the supplied shape.
     * @throws IllegalArgumentException if the shape is larger than the largest buffer.
     * @param shape The shape.
     * @return A long buffer.
     */
    private static LongBuffer alloc(long[] shape) {
        int elements = computeNumElements(shape);
        if (elements < 0) {
            throw new IllegalArgumentException("Invalid shape for Java tensor, expected less than Integer.MAX_VALUE elements, found " + Arrays.toString(shape));
        }
        return ByteBuffer.allocateDirect(elements * Long.BYTES).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer();
    }
}
