/*
 * Copyright (c) 2023, 2025, Oracle and/or its affiliates. All rights reserved.
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

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensorLike;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Interface for input processors for embedding models. The input processor deals with tokenization, padding and truncation.
 */
public interface InputProcessor extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Validates that the model is compatible with this input processor.
     * @param inputInfo The input information from the ONNX model.
     * @return True if the ONNX model is compatible with this processor.
     */
    boolean validate(Map<String, NodeInfo> inputInfo);

    /**
     * The vocabulary understood by this input processor.
     * @return The vocab set.
     */
    Set<String> getVocab();

    /**
     * The token id for the supplied string, or the id of the unknown token.
     * @param token The token to look up.
     * @return The id for the supplied token.
     */
    long getTokenId(String token);

    /**
     * The max length of inputs that this processor (and model) supports.
     * @return The maximum input length in tokens.
     */
    int getMaxLength();

    /**
     * Processes the input strings into an input record.
     * @param env The ONNX Runtime environment so it can construct the appropriate tensors.
     * @param input The strings to embed.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    ProcessedInput process(OrtEnvironment env, List<String> input)
        throws OrtException;

    /**
     * Processes the input string into an input record.
     * @param env The ONNX Runtime environment so it can construct the appropriate tensors.
     * @param input The string to embed.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    default ProcessedInput process(OrtEnvironment env, String input) throws OrtException {
        return process(env, List.of(input));
    }

    /**
     * Processes the input strings into an input record.
     * <p>
     * Throws {@link IllegalArgumentException} if the cache is not large enough to store the input batch.
     * @param env The ONNX Runtime environment.
     * @param input The strings to embed.
     * @param cache An input cache containing direct byte buffers.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    ProcessedInput process(OrtEnvironment env, List<String> input, BufferCache cache) throws OrtException;

    /**
     * Processes the input string into an input record.
     * <p>
     * Throws {@link IllegalArgumentException} if the cache is not large enough to store the input batch.
     * @param env The ONNX Runtime environment.
     * @param input The string to embed.
     * @param cache An input cache containing direct byte buffers.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    default ProcessedInput process(OrtEnvironment env, String input, BufferCache cache) throws OrtException {
        return process(env, List.of(input), cache);
    }

    /**
     * Processes the input tokens into an input record.
     * @param env The ONNX Runtime environment so it can construct the appropriate tensors.
     * @param tokens The pre-tokenized input to embed.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    ProcessedInput processTokensBatch(OrtEnvironment env, List<List<String>> tokens)
            throws OrtException;

    /**
     * Processes the input tokens into an input record.
     * @param env The ONNX Runtime environment so it can construct the appropriate tensors.
     * @param tokens The pre-tokenized input to embed.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    default ProcessedInput processTokens(OrtEnvironment env, List<String> tokens) throws OrtException {
        return processTokensBatch(env, List.of(tokens));
    }

    /**
     * Processes the input tokens into an input record.
     * <p>
     * Throws {@link IllegalArgumentException} if the cache is not large enough to store the input batch.
     * @param env The ONNX Runtime environment.
     * @param tokens The pre-tokenized input to embed.
     * @param cache An input cache containing direct byte buffers.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    ProcessedInput processTokensBatch(OrtEnvironment env, List<List<String>> tokens, BufferCache cache) throws OrtException;

    /**
     * Processes the input tokens into an input record.
     * <p>
     * Throws {@link IllegalArgumentException} if the cache is not large enough to store the input batch.
     * @param env The ONNX Runtime environment.
     * @param tokens The pre-tokenized input to embed.
     * @param cache An input cache containing direct byte buffers.
     * @return A processed input record.
     * @throws OrtException If tensor creation failed.
     */
    default ProcessedInput processTokens(OrtEnvironment env, List<String> tokens, BufferCache cache) throws OrtException {
        return processTokensBatch(env, List.of(tokens), cache);
    }

    /**
     * Constructs a buffer cache instance that contains all the necessary buffers for this input processor.
     * <p>
     * {@code batchSize} * {@code numTokens} must be less than {@code 2^31}.
     * @param maxBatchSize The maximum batch size.
     * @param maxNumTokens The maximum number of tokens.
     * @return An input cache.
     */
    public BufferCache createInputCache(int maxBatchSize, int maxNumTokens);

    /**
     * A record containing the inputs ready for ORT along with the example lengths before padding.
     * @param inputs The ORT inputs.
     * @param tokenLengths The example lengths before padding.
     * @param tokenIds The token ids including padding.
     */
    public record ProcessedInput(Map<String, ? extends OnnxTensorLike> inputs,
                                 long[] tokenLengths,
                                 LongTensorBuffer tokenIds) implements AutoCloseable {
        @Override
        public void close() {
            OnnxValue.close(inputs);
        }
    }

}
