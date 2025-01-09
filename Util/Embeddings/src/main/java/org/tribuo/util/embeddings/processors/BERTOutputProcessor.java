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
import org.tribuo.util.embeddings.FloatTensorBuffer;
import org.tribuo.util.embeddings.OutputProcessor;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.logging.Logger;

/**
 * An output processor which operates on BERT style models which output "last_hidden_state" or its equivalent which
 * contains all the token embeddings. It can optionally emit a pooled output state using the pooler built into
 * the model.
 */
public final class BERTOutputProcessor implements OutputProcessor {
    private static final Logger logger = Logger.getLogger(BERTOutputProcessor.class.getName());

    /**
     * The type of output pooling to perform.
     */
    public enum BERTPooling {
        /**
         * Returns the pooled output.
         */
        POOLER,
        /**
         * Returns the CLS embedding.
         */
        CLS,
        /**
         * Takes the average of the token embeddings
         */
        MEAN,
        /**
         * Returns the token embeddings.
         */
        TOKEN;
    }

    // BERT output names
    /**
     * Output name for the token level outputs.
     */
    public static final String POOLED_OUTPUT = "pooler_output";
    /**
     * Output name for the token level outputs.
     */
    public static final String TOKEN_OUTPUT = "last_hidden_state";

    /**
     * Size of the embedding dimension.
     */
    @Config(mandatory = true, description = "Size of the embedding dimension.")
    private int embeddingDimension;

    /**
     * Type of pooling to use when returning a single embedding for the input sequence.
     */
    @Config(description="Type of pooling to use when returning a single embedding for the input sequence")
    private BERTPooling pooling = BERTPooling.CLS;

    /**
     * Use all tokens to compute the aggregated token vector including [CLS] and [SEP]?
     */
    @Config(description = "Use all tokens to compute the aggregated token vector including [CLS] and [SEP]?")
    private boolean useAllTokens = false;

    /**
     * L2 normalize the output?
     */
    @Config(description = "L2 normalize the output")
    private boolean normalize = false;

    /**
     * Pooler output name.
     */
    @Config(description = "Pooler output name")
    private String poolerOutput = POOLED_OUTPUT;

    /**
     * Output name.
     */
    @Config(description = "Output name")
    private String tokenOutput = TOKEN_OUTPUT;

    /*
     * Will be set to true if the embedding size is left unbound in the model.  When this
     * variable is true, each output embedding needs to be checked to confirm it is of
     * the expected dimensionality (i.e. {@link #embeddingDimension}).
     */
    private boolean unspecifiedEmbeddingDimension = false;

    /**
     * For OLCUT.
     */
    private BERTOutputProcessor() {}

    /**
     * Constructs a BERTOutputProcessor using the supplied arguments.
     * @param pooling The type of pooling operation to apply to the model output.
     * @param embeddingDimension The embedding dimension.
     * @param normalize Should the output be normalized into a unit vector?
     * @param useAllTokens Should the output include the BOS and EOS tokens if it uses average pooling?
     * @param outputName The name of the model output to process.
     */
    public BERTOutputProcessor(BERTPooling pooling, int embeddingDimension, boolean normalize, boolean useAllTokens, String outputName) {
        this.pooling = pooling;
        this.embeddingDimension = embeddingDimension;
        this.normalize = normalize;
        this.useAllTokens = useAllTokens;
        if (pooling == BERTPooling.POOLER) {
            this.poolerOutput = outputName;
        } else {
            this.tokenOutput = outputName;
        }
    }

    @Override
    public boolean validate(Map<String, NodeInfo> outputs) {
        if (outputs.size() > 2) {
            logger.warning("Invalid model, expected 1 or 2 outputs, found " + outputs.size());
            return false;
        } else {
            return switch (pooling) {
                case POOLER: {
                    // check that the output has the expected name
                    NodeInfo outputPooler = outputs.get(poolerOutput);
                    if (outputPooler.getInfo() instanceof TensorInfo outputPoolerTensor) {
                        long[] shape = outputPoolerTensor.getShape();
                        if (shape.length != 2) {
                            logger.warning("Invalid model, expected to find '" + poolerOutput
                                + "' with 2 dimensions, found :" + Arrays.toString(shape));
                            yield false;
                        } else if (embeddingDimension != (int) shape[1]) {
                            // dimensions should be [batch_size, embedding_dim]
                            if ((int) shape[1] == -1) {
                                logger.warning("Model does not specify embedding dimension. Provided "
                                        + "size of " + embeddingDimension + " will be checked for each output "
                                        + "at inference time");
                                unspecifiedEmbeddingDimension = true;
                                yield true;
                            } else {
                                logger.warning(
                                        "Invalid model, expected to find embedding dimension of "
                                                + embeddingDimension + " but found " + shape[1]);
                                yield false;
                            }
                        } else {
                            yield true;
                        }
                    } else {
                        logger.warning(
                            "Invalid model, expected to find tensor output called '" + poolerOutput
                                + "'");
                        yield false;
                    }
                }
                case CLS:
                case MEAN:
                case TOKEN: {
                    // check that the output has the expected name
                    NodeInfo outputToken = outputs.get(tokenOutput);
                    if (outputToken.getInfo() instanceof TensorInfo outputTokenTensor) {
                        long[] shape = outputTokenTensor.getShape();
                        if (shape.length != 3) {
                            logger.warning("Invalid model, expected to find '" + tokenOutput
                                + "' with 3 dimensions, found :" + Arrays.toString(shape));
                            yield false;
                        } else if (embeddingDimension != (int) shape[2]) {
                            // Bert embedding dim is the last dimension.
                            // The first two are batch and sequence length.
                            if ((int)shape[2] == -1) {
                                logger.warning("Model does not specify embedding dimension. Provided "
                                + "size of " + embeddingDimension + " will be checked for each output "
                                + "at inference time");
                                unspecifiedEmbeddingDimension = true;
                                yield true;
                            } else {
                                logger.warning(
                                        "Invalid model, expected to find embedding dimension of "
                                                + embeddingDimension + " but found " + shape[2]);
                                yield false;
                            }
                        } else {
                            yield true;
                        }
                    } else {
                        logger.warning(
                            "Invalid model, expected to find tensor output called '" + tokenOutput
                                + "'");
                        yield false;
                    }
                }
            };
        }
    }

    @Override
    public int getEmbeddingDimension() {
        return embeddingDimension;
    }

    @Override
    public Map<String, FloatTensorBuffer> process(Result result, long[] inputLengths) {
        OnnxTensor value = (OnnxTensor) result.get(tokenOutput).orElseThrow(() -> new IllegalStateException("Failed to read " + tokenOutput + " from the BERT response"));
        FloatTensorBuffer featureValues = switch (pooling) {
            case POOLER -> extractPooledVector((OnnxTensor)result.get(poolerOutput).orElseThrow(() -> new IllegalStateException("Failed to read " + poolerOutput + " from the BERT response")));
            case CLS -> extractCLSVector(value);
            case MEAN -> extractPooledTokenVector(value, inputLengths);
            case TOKEN -> extractTokens(value);
        };
        if (normalize) {
            featureValues.l2InPlace();
        }
        return Collections.singletonMap(tokenOutput, featureValues);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "OutputProcessor");
    }

    /**
     * Extracts the pooled vector from the session output.
     * <p>
     * Throws IllegalStateException if the session output didn't parse.
     * @param tensor The hidden state tensor.
     * @return The pooler output vector in a float buffer.
     */
    private FloatTensorBuffer extractPooledVector(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            long[] shape = tensor.getInfo().getShape();
            if (unspecifiedEmbeddingDimension && embeddingDimension != shape[2]) {
                throw new IllegalStateException("Expected embedding dimension " + embeddingDimension
                        + " but found " + shape[2]);
            }
            FloatTensorBuffer output = new FloatTensorBuffer(new long[]{shape[0],embeddingDimension}, false);
            output.buffer().put(buffer);
            output.buffer().rewind();
            return output;
        } else {
            throw new IllegalStateException("Expected a float tensor, found " + tensor.getInfo().toString());
        }
    }

    /**
     * Extracts the CLS vector from the session output.
     * <p>
     * Throws IllegalStateException if the session output didn't parse.
     * @param tensor The hidden state tensor.
     * @return The cls vector in a float buffer.
     */
    private FloatTensorBuffer extractCLSVector(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            long[] shape = tensor.getInfo().getShape();
            if (unspecifiedEmbeddingDimension && (shape[2] != embeddingDimension)) {
                throw new IllegalStateException("Expecting embedding dimension " + embeddingDimension
                        + " but found " + shape[2]);
            }
            FloatTensorBuffer output = new FloatTensorBuffer(new long[]{shape[0],embeddingDimension}, false);
            for (int i = 0; i < (int) shape[0]; i++) {
                int inputOffset = (int) (i * shape[1] * shape[2]);
                int outputOffset = i * embeddingDimension;
                output.buffer().put(outputOffset, buffer, inputOffset, embeddingDimension);
            }
            return output;
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
    private FloatTensorBuffer extractPooledTokenVector(OnnxTensor tensor, long[] inputLengths) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            long[] shape = tensor.getInfo().getShape();
            if (unspecifiedEmbeddingDimension && (shape[2] != embeddingDimension)) {
                throw new IllegalStateException("Expecting embedding dimension " + embeddingDimension
                        + " but found " + shape[2]);
            }
            FloatTensorBuffer output = new FloatTensorBuffer(new long[]{inputLengths.length, embeddingDimension}, false);
            for (int i = 0; i < inputLengths.length; i++) {
                int offset = embeddingDimension * (int)shape[1] * i;
                int outputOffset = embeddingDimension * i;
                int startPos = useAllTokens ? 0 : 1;
                int endPos = (int) inputLengths[i] - (useAllTokens ? 0 : 1);
                // iterate the tokens, creating new examples
                for (int j = startPos; j < endPos; j++) {
                    for (int k = 0; k < embeddingDimension; k++) {
                        float cur = output.buffer().get(outputOffset + k);
                        int idx = offset + (j * embeddingDimension) + k;
                        if (idx >= buffer.capacity()) {
                            throw new IndexOutOfBoundsException("Index " + idx + " is out of bounds for capacity " + buffer.capacity());
                        }
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

    private FloatTensorBuffer extractTokens(OnnxTensor tensor) {
        FloatBuffer buffer = tensor.getFloatBuffer();
        if (buffer != null) {
            long[] shape = tensor.getInfo().getShape();
            if (unspecifiedEmbeddingDimension && embeddingDimension != shape[2]) {
                throw new IllegalStateException("Expected embedding dimension " + embeddingDimension
                        + " but found " + shape[2]);
            }
            return new FloatTensorBuffer(buffer, tensor.getInfo().getShape());
        } else {
            throw new IllegalStateException("Expected a float tensor, found " + tensor.getInfo().toString());
        }
    }
}
