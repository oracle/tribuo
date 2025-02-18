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
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import java.util.Map;

/**
 * Interface for processing the outputs of a model execution into tensor buffers.
 */
public interface OutputProcessor extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Validates that the model and output processor are compatible.
     * @param outputInfo The output information from the ONNX model.
     * @return True if the model produces compatible outputs.
     */
    boolean validate(Map<String, NodeInfo> outputInfo);

    /**
     * Returns the embedding dimension supported by this output processor.
     * @return The embedding dimension.
     */
    int getEmbeddingDimension();

    /**
     * Processes the session result into a map of tensor buffers suitable for downstream processing.
     * <p>
     * Does not close the {@code result} object.
     * @param result The session output.
     * @param inputLengths The lengths of each input element.
     * @return The map of tensor buffers.
     */
    Map<String, FloatTensorBuffer> process(OrtSession.Result result, long[] inputLengths);

    /**
     * Create the output cache for this output processor.
     * @param maxBatchSize The maximum batch size.
     * @param maxNumTokens The maximum number of tokens.
     * @return The output cache.
     */
    BufferCache createOutputCache(int maxBatchSize, int maxNumTokens);

    /**
     * Creates the output tensors from the supplied cache and parameters.
     * @param env The OrtEnvironment which will own the tensors.
     * @param cache The output cache.
     * @param batchSize The batch size.
     * @param numTokens The number of tokens.
     * @return The prepared output tensors.
     * @throws OrtException If the tensors could not be created.
     */
    Map<String, OnnxTensor> createOutputTensors(OrtEnvironment env, BufferCache cache, int batchSize, int numTokens) throws OrtException;
}
