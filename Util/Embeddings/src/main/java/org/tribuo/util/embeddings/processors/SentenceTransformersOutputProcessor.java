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

package org.tribuo.util.embeddings.processors;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.TensorInfo;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.embeddings.FloatTensor;
import org.tribuo.util.embeddings.OutputProcessor;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.logging.Logger;

/**
 *
 */
public final class SentenceTransformersOutputProcessor implements OutputProcessor {
    private static final Logger logger = Logger.getLogger(SentenceTransformersOutputProcessor.class.getName());

    /**
     * The type of output pooling to perform.
     */
    public enum SentencePooling {
        /**
         * Returns the sentence embedding.
         */
        SENTENCE,
        /**
         * Takes the average of the token embeddings
         */
        MEAN_TOKEN,
        /**
         * Returns the token embeddings.
         */
        TOKENS;
    }

    // Sentence transformers output names
    /**
     * Output name for the token level outputs.
     */
    public static final String TOKEN_OUTPUT = "token_embeddings";
    /**
     * Output name for the sentence level outputs.
     */
    public static final String SENTENCE_OUTPUT = "sentence_embedding";

    @Config(mandatory = true, description = "Size of the embedding dimension.")
    private int embeddingDimension;

    @Config(description="Type of pooling to use when returning a single embedding for the input sequence")
    private SentencePooling pooling = SentencePooling.SENTENCE;

    @Config(description = "Use all tokens to compute the aggregated token vector including [CLS] and [SEP]?")
    private boolean useAllTokens = false;

    @Config(description = "L2 normalize the output")
    private boolean normalize = false;

    @Config(description = "Token output name")
    private String tokenOutput = TOKEN_OUTPUT;

    @Config(description = "Sentence output name")
    private String sentenceOutput = SENTENCE_OUTPUT;

    /**
     * For OLCUT.
     */
    private SentenceTransformersOutputProcessor() {}

    public SentenceTransformersOutputProcessor(SentencePooling pooling, int embeddingDimension, boolean normalize, boolean useAllTokens, String tokenOutput, String sentenceOutput) {
        this.pooling = pooling;
        this.embeddingDimension = embeddingDimension;
        this.normalize = normalize;
        this.useAllTokens = useAllTokens;
        this.tokenOutput = tokenOutput;
        this.sentenceOutput = sentenceOutput;
    }

    @Override
    public boolean validate(Map<String, NodeInfo> outputs) {
        if (outputs.size() != 2) {
            logger.warning("Invalid model, expected 2 outputs, found " + outputs.size());
            return false;
        } else {
            // check that the outputs have the expected names
            NodeInfo outputZero = outputs.get(tokenOutput);
            if (outputZero.getInfo() instanceof TensorInfo outputZeroTensor) {
                long[] shape = outputZeroTensor.getShape();
                if (shape.length != 3) {
                    logger.warning("Invalid model, expected to find '" + tokenOutput + "' with 3 dimensions, found :" + Arrays.toString(shape));
                    return false;
                } else {
                    // Bert embedding dim is the last dimension.
                    // The first two are batch and sequence length.
                    if (embeddingDimension != (int) shape[2]) {
                        logger.warning("Invalid model, expected to find embedding dimension of " + embeddingDimension + " but found " + shape[2]);
                        return false;
                    }
                }
            } else {
                logger.warning("Invalid model, expected to find tensor output called '" + tokenOutput + "'");
                return false;
            }
            NodeInfo outputOne = outputs.get(sentenceOutput);
            if (outputOne.getInfo() instanceof TensorInfo outputOneTensor) {
                long[] shape = outputOneTensor.getShape();
                if (shape.length != 2) {
                    logger.warning("Invalid model, expected to find '" + sentenceOutput + "' with 2 dimensions, found :" + Arrays.toString(shape));
                    return false;
                } else {
                    // Bert embedding dim is the last dimension.
                    // The first one is batch.
                    if (embeddingDimension != (int) shape[1]) {
                        logger.warning("Invalid model, expected to find embedding dimension of " + embeddingDimension + " but found " + shape[1]);
                        return false;
                    } else {
                        return true;
                    }
                }
            } else {
                logger.warning("Invalid model, expected to find tensor output called '" + sentenceOutput + "'");
                return false;
            }
        }
    }

    @Override
    public int getEmbeddingDimension() {
        return embeddingDimension;
    }

    @Override
    public Map<String, FloatTensor> process(Result result, long[] inputLengths) {
        OnnxTensor sentenceVectors = (OnnxTensor) result.get(sentenceOutput).orElseThrow(() -> new IllegalStateException("Failed to read " + sentenceOutput + " from the model response"));
        OnnxTensor tokenVectors = (OnnxTensor) result.get(tokenOutput).orElseThrow(() -> new IllegalStateException("Failed to read " + tokenOutput + " from the model response"));
        String keyName = tokenOutput;
        FloatTensor featureValues = switch (pooling) {
            case SENTENCE -> { keyName=sentenceOutput; yield extractCLSVector(sentenceVectors); }
            case MEAN_TOKEN -> extractPooledTokenVector(tokenVectors, inputLengths);
            case TOKENS -> extractTokens(tokenVectors);
        };
        if (normalize) {
            featureValues.l2InPlace();
        }
        return Collections.singletonMap(keyName, featureValues);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "OutputProcessor");
    }

    /**
     * Extracts the CLS vector from the session output.
     * <p>
     * Throws IllegalStateException if the session output didn't parse.
     * @param tensor The hidden state tensor.
     * @return The cls vector in a float buffer.
     */
    private FloatTensor extractCLSVector(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            return new FloatTensor(buffer, tensor.getInfo().getShape());
        } else {
            throw new IllegalStateException("Expected a float tensor, found " + tensor.getInfo().toString());
        }
    }

    /**
     * Extracts the token level outputs, averaging or summing them into a single FloatBuffer.
     * <p>
     * Throws IllegalStateException if the session output didn't parse.
     * @param tensor The hidden state tensor.
     * @param inputLengths The lengths of the unpadded input.
     * @return The aggregated token embeddings as a double array.
     */
    private FloatTensor extractPooledTokenVector(OnnxTensor tensor, long[] inputLengths) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            long[] shape = tensor.getInfo().getShape();
            FloatTensor output = new FloatTensor(new long[]{inputLengths.length, embeddingDimension}, false);
            for (int i = 0; i < inputLengths.length; i++) {
                int offset = embeddingDimension * (int)shape[1] * i;
                int outputOffset = embeddingDimension * i;
                int startPos = useAllTokens ? 0 : 1;
                int endPos = (int) inputLengths[i] + (useAllTokens ? 2 : 1);
                // iterate the tokens, creating new examples
                for (int j = startPos; j < endPos; j++) {
                    for (int k = 0; k < embeddingDimension; k++) {
                        float cur = output.get(outputOffset + k);
                        float update = buffer.get(offset + (j * embeddingDimension) + k);
                        output.buffer().put(outputOffset + k, cur + update);
                    }
                }
                for (int j = 0; j < embeddingDimension; j++) {
                    float tmp = output.buffer().get(outputOffset + j);
                    output.buffer().put(outputOffset + j, tmp / inputLengths[i]);
                }
            }
            return output;
        } else {
            throw new IllegalStateException("Expected a float tensor, found " + tensor.getInfo().toString());
        }
    }

    private FloatTensor extractTokens(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            return new FloatTensor(buffer, tensor.getInfo().getShape());
        } else {
            throw new IllegalStateException("Expected a float tensor, found " + tensor.getInfo().toString());
        }
    }
}
