/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.interop.onnx.protos.ExampleTransformerProto;
import org.tribuo.math.la.SparseVector;
import org.tribuo.protos.ProtoSerializable;

import java.io.Serializable;
import java.util.List;

/**
 * Transforms a {@link SparseVector}, extracting the features from it as a {@link OnnxTensor}.
 * <p>
 * This usually densifies the example, so can be a lot larger than the input example.
 * <p>
 * N.B. ONNX support is experimental, and may change without a major version bump.
 */
public interface ExampleTransformer extends Configurable, ProtoSerializable<ExampleTransformerProto>, Provenancable<ConfiguredObjectProvenance>, Serializable {

    /**
     * Converts a {@link SparseVector} representing the features into a {@link OnnxTensor}.
     * <p>
     * It generates it as a single example minibatch.
     * @param env The OrtEnvironment to create the tensor in.
     * @param vector The features to convert.
     * @return A dense OnnxTensor representing this vector.
     * @throws OrtException if the transformation failed.
     */
    public OnnxTensor transform(OrtEnvironment env, SparseVector vector) throws OrtException;

    /**
     * Converts a list of {@link SparseVector}s representing a batch of features into a {@link OnnxTensor}.
     *
     * @param env The OrtEnvironment to create the tensor in.
     * @param vectors The batch of features to convert.
     * @return A dense OnnxTensor representing this minibatch.
     * @throws OrtException if the transformation failed.
     */
    public OnnxTensor transform(OrtEnvironment env, List<SparseVector> vectors) throws OrtException;

}
