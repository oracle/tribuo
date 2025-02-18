/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

import ai.onnxruntime.OnnxJavaType;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A cache which holds the buffers required to store the model's inputs and outputs.
 * <p>
 * Used to reduce byte buffer allocations in high-throughput systems. Typically there will be one
 * cache object for the inputs and one for the outputs for each thread that executes the model.
 * These caches should be reused within their threads, and the buffers they contain are not thread safe.
 */
public final class BufferCache {
    /**
     * Maximum size in bytes of a cache buffer.
     */
    public static final int MAX_SIZE = 1 << 30;

    private final Map<String, Buffer> cache;
    private final Map<String, TensorDescription> descriptions;

    private final int maxBatchSize;
    private final int maxNumTokens;
    private final int embeddingDim;

    /**
     * Build a buffer cache from a list of tensor descriptions and the sizes.
     * @param descriptions The tensor descriptions.
     * @param maxBatchSize The maximum batch size supported by this cache.
     * @param maxNumTokens The maximum number of tokens supported by this cache (typically the max tokens supported by the model).
     */
    public BufferCache(List<TensorDescription> descriptions, int maxBatchSize, int maxNumTokens) {
        this(descriptions, maxBatchSize, maxNumTokens, 1);
        for (var d : descriptions) {
            if (d.shapeType() == Shape.BATCH_EMBED || d.shapeType() == Shape.BATCH_TOKEN_EMBED) {
                throw new IllegalArgumentException("Invalid tensor description for this constructor, must not require the embedding dimension.");
            }
        }
    }

    /**
     * Build a buffer cache from a list of tensor descriptions and the sizes.
     * @param descriptions The tensor descriptions.
     * @param maxBatchSize The maximum batch size supported by this cache.
     * @param maxNumTokens The maximum number of tokens supported by this cache (typically the max tokens supported by the model).
     * @param embeddingDim The embedding dimension.
     */
    public BufferCache(List<TensorDescription> descriptions, int maxBatchSize, int maxNumTokens, int embeddingDim) {
        if (maxBatchSize < 1) {
            throw new IllegalArgumentException("Invalid maximum batch size, must be positive, found " + maxBatchSize);
        }
        if (maxNumTokens < 1) {
            throw new IllegalArgumentException("Invalid maximum number of tokens, must be positive, found " + maxNumTokens);
        }
        if (embeddingDim < 1) {
            throw new IllegalArgumentException("Invalid embedding dimension, must be positive, found " + embeddingDim);
        }
        this.maxBatchSize = maxBatchSize;
        this.maxNumTokens = maxNumTokens;
        this.embeddingDim = embeddingDim;
        Map<String, Buffer> buffers = new HashMap<>();
        Map<String, TensorDescription> descMap = new HashMap<>();
        for (var desc : descriptions) {
            var tmp = descMap.put(desc.name(), desc);
            if (tmp != null) {
                throw new IllegalArgumentException("Invalid description list, duplicate entry for name " + desc.name());
            }
            buffers.put(desc.name(), desc.createBuffer(maxBatchSize, maxNumTokens, embeddingDim));
            switch (desc.shapeType) {
                case BATCH_TOKEN_EMBED -> {
                    if (desc.type.size * maxBatchSize * maxNumTokens * embeddingDim > BufferCache.MAX_SIZE) {
                        throw new IllegalArgumentException("Invalid cache size requested, must be less than 2^31 bytes");
                    }
                }
                case BATCH_EMBED -> {
                    if (desc.type.size * maxBatchSize * embeddingDim > BufferCache.MAX_SIZE) {
                        throw new IllegalArgumentException("Invalid cache size requested, must be less than 2^31 bytes");
                    }
                }
                case BATCH_TOKEN -> {
                    if (desc.type.size * maxBatchSize * maxNumTokens > BufferCache.MAX_SIZE) {
                        throw new IllegalArgumentException("Invalid cache size requested, must be less than 2^31 bytes, maxBatchSize " + maxBatchSize + ", maxNumTokens " + maxNumTokens);
                    }
                }
                case BATCH -> {
                    // seems pretty unlikely people would request a maxBatchSize of Integer.MAX_INT.
                    if (desc.type.size * maxBatchSize > BufferCache.MAX_SIZE) {
                        throw new IllegalArgumentException("Invalid cache size requested, must be less than 2^31 bytes");
                    }
                }
            }
        }
        this.cache = Collections.unmodifiableMap(buffers);
        this.descriptions = Collections.unmodifiableMap(descMap);
    }

    /**
     * The maximum batch size supported by this buffer cache.
     * @return The maximum batch size.
     */
    public int maxBatchSize() {
        return maxBatchSize;
    }

    /**
     * The maximum number of tokens supported by this buffer cache.
     * @return The maximum number of tokens.
     */
    public int maxNumTokens() {
        return maxNumTokens;
    }

    /**
     * The embedding dimension.
     * @return The embedding dimension.
     */
    public int embeddingDim() {
        return embeddingDim;
    }

    /**
     * The number of buffers in this cache.
     * @return The number of buffers.
     */
    public int numBuffers() {
        return cache.size();
    }

    /**
     * Gets the named buffer.
     * @param name The buffer name.
     * @return Optional containing the buffer or empty if the name is unknown.
     */
    public Optional<Buffer> get(String name) {
        return Optional.ofNullable(cache.get(name));
    }

    /**
     * Returns a zero based slice of the named buffer of the requested size.
     * @param name The buffer name.
     * @param size The slice size.
     * @return Optional containing the buffer or empty if the name is unknown.
     */
    public Optional<Buffer> slice(String name, int size) {
        var buf = cache.get(name);
        if (buf != null) {
            return Optional.of(buf.slice(0, size));
        } else {
            return Optional.empty();
        }
    }

    /**
     * Returns a zero based slice of the named buffer of the requested size.
     * <p>
     * Throws {@link IllegalArgumentException} if the name is unknown.
     * @param name The buffer name.
     * @param size The slice size.
     * @return The sliced buffer.
     */
    public Buffer sliceOrThrow(String name, int size) {
        var buf = cache.get(name);
        if (buf != null) {
            return buf.slice(0, size);
        } else {
            throw new IllegalArgumentException("Expected buffer named '" + name + "'");
        }
    }

    /**
     * Gets the named buffer description.
     * @param name The buffer name.
     * @return Optional containing the buffer description or empty if the name is unknown.
     */
    public Optional<TensorDescription> getDescription(String name) {
        return Optional.ofNullable(descriptions.get(name));
    }

    /**
     * Shape of the tensor.
     */
    public enum Shape {
        /**
         * [batch_size, num_tokens, embedding_dimension]
         */
        BATCH_TOKEN_EMBED,
        /**
         * [batch_size, embedding_dimension]
         */
        BATCH_EMBED,
        /**
         * [batch_size, num_tokens]
         */
        BATCH_TOKEN,
        /**
         * [batch_size]
         */
        BATCH;
    }

    /**
     * Description of output tensors used to build the {@link BufferCache}.
     * @param name The name of the output.
     * @param shapeType The symbolic shape.
     * @param type The type of the buffer.
     */
    public record TensorDescription(String name, Shape shapeType, OnnxJavaType type) {
        /**
         * Creates the appropriately sized buffer object based on this description.
         * @return The buffer object.
         */
        Buffer createBuffer(int maxBatchSize, int maxNumTokens, int embeddingDim) {
            int byteSize = type.size * maxBatchSize * maxNumTokens * embeddingDim;
            ByteBuffer buf = ByteBuffer.allocateDirect(byteSize).order(ByteOrder.nativeOrder());
            return switch (type) {
                case FLOAT -> buf.asFloatBuffer();
                case DOUBLE -> buf.asDoubleBuffer();
                case UINT8, INT8 -> buf;
                case INT16, FLOAT16, BFLOAT16 -> buf.asShortBuffer();
                case INT32 -> buf.asIntBuffer();
                case INT64 -> buf.asLongBuffer();
                default -> throw new IllegalStateException("Invalid types for buffer creation.");
            };
        }
    }
}
