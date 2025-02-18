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

package org.tribuo.util.embeddings.processors;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.TensorInfo;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.embeddings.BufferCache;
import org.tribuo.util.embeddings.FloatTensorBuffer;
import org.tribuo.util.embeddings.OutputProcessor;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * A no-op output processor that returns a single output from the model as a float tensor without any further processing.
 */
public final class NoOpOutputProcessor implements OutputProcessor {
    private static final Logger logger = Logger.getLogger(NoOpOutputProcessor.class.getName());

    public static final String DEFAULT_OUTPUT_NAME = "embedding";

    /**
     * Size of the embedding dimension.
     */
    @Config(mandatory = true, description = "Size of the embedding dimension.")
    private int embeddingDimension;

    /**
     * Output name.
     */
    @Config(description = "Output name")
    private String outputName = DEFAULT_OUTPUT_NAME;

    /**
     * For OLCUT.
     */
    private NoOpOutputProcessor() {}

    /**
     * Constructs a NoOpOutputProcessor with the default name.
     * @param embeddingDimension The embedding dimension.
     */
    public NoOpOutputProcessor(int embeddingDimension) {
        this(embeddingDimension, DEFAULT_OUTPUT_NAME);
    }

    /**
     * Constructs a NoOpOutputProcessor with the specified name and output dimension.
     * @param embeddingDimension The embedding dimension.
     * @param outputName The output name.
     */
    public NoOpOutputProcessor(int embeddingDimension, String outputName) {
        this.embeddingDimension = embeddingDimension;
        this.outputName = outputName;
    }

    @Override
    public boolean validate(Map<String, NodeInfo> outputs) {
        if (outputs.size() != 1) {
            logger.warning("Invalid model, expected 1 outputs, found " + outputs.size());
            return false;
        } else {
            // check that the output has the expected name
            NodeInfo outputZero = outputs.get(outputName);
            if (outputZero.getInfo() instanceof TensorInfo outputZeroTensor) {
                long[] shape = outputZeroTensor.getShape();
                if (shape.length != 2) {
                    logger.warning("Invalid model, expected to find '" + outputName + "' with 2 dimensions, found :" + Arrays.toString(shape));
                    return false;
                } else {
                    // shape should be [batch_size, embedding_dim]
                    if (embeddingDimension != (int) shape[1]) {
                        logger.warning("Invalid model, expected to find embedding dimension of " + embeddingDimension + " but found " + shape[1]);
                        return false;
                    } else {
                        return true;
                    }
                }
            } else {
                logger.warning("Invalid model, expected to find tensor output called '" + outputName + "'");
                return false;
            }
        }
    }

    @Override
    public int getEmbeddingDimension() {
        return embeddingDimension;
    }

    @Override
    public Map<String, FloatTensorBuffer> process(Result result, long[] inputLengths) {
        Map<String, FloatTensorBuffer> outputs = new HashMap<>(result.size());
        result.forEach(resultEntry -> {
            OnnxTensor tensor = (OnnxTensor) resultEntry.getValue();
            FloatBuffer buffer = tensor.getFloatBuffer();
            if (buffer != null) {
                outputs.put(resultEntry.getKey(), new FloatTensorBuffer(buffer, tensor.getInfo().getShape()));
            } else {
                throw new IllegalStateException("Expected a float tensor, found " + tensor.getInfo().toString());
            }
        });
        return outputs;
    }

    @Override
    public BufferCache createOutputCache(int maxBatchSize, int maxNumTokens) {
        List<BufferCache.TensorDescription> descriptions = List.of(
                new BufferCache.TensorDescription(outputName, BufferCache.Shape.BATCH_EMBED, OnnxJavaType.FLOAT)
        );
        return new BufferCache(descriptions, maxBatchSize, maxNumTokens, embeddingDimension);
    }

    @Override
    public Map<String, OnnxTensor> createOutputTensors(OrtEnvironment env, BufferCache cache, int batchSize, int numTokens) throws OrtException {
        int outputSize = batchSize * embeddingDimension;
        var outputTensor = new FloatTensorBuffer((FloatBuffer)cache.sliceOrThrow(outputName, outputSize), new long[]{batchSize, embeddingDimension}, 0.0f);
        return Map.of(outputName, outputTensor.wrapForORT(env));
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "OutputProcessor");
    }
}
