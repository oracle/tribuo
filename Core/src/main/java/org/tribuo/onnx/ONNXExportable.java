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

package org.tribuo.onnx;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.config.protobuf.ProtoProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.io.ProvenanceSerialization;
import org.tribuo.provenance.ModelProvenance;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Path;

/**
 * An interface which denotes this {@link org.tribuo.Model} can be
 * exported as an ONNX model.
 * <p>
 * Tribuo models export with a single input of size [-1, numFeatures] and a
 * single output of size [-1, numOutputDimensions]. The first dimension in both
 * is defined to be an unbound dimension called "batch", which denotes the batch size.
 */
public interface ONNXExportable {

    /**
     * The provenance serializer.
     */
    public static final ProvenanceSerialization SERIALIZER = new ProtoProvenanceSerialization(true);

    /**
     * The name of the ONNX metadata field where the provenance information is stored
     * in exported models.
     */
    public static final String PROVENANCE_METADATA_FIELD = "TRIBUO_PROVENANCE";

    /**
     * Exports this {@link org.tribuo.Model} as an ONNX protobuf.
     * @param domain A reverse-DNS name to namespace the model (e.g., org.tribuo.classification.sgd.linear).
     * @param modelVersion A version number for this model.
     * @return The ONNX ModelProto representing this Tribuo Model.
     */
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion);

    /**
     * Writes this {@link org.tribuo.Model} into the {@link OnnxMl.GraphProto.Builder} inside the {@link ONNXContext}.
     * @param context The ONNX context which stores the builder and namespace.
     * @param inputName The name the model should take its input from.
     * @param outputName The name the model should output to.
     */
    public void writeONNXGraph(ONNXContext context, String inputName, String outputName);

    /**
     * Exports this {@link org.tribuo.Model} as an ONNX file.
     * @param domain A reverse-DNS name to namespace the model (e.g., org.tribuo.classification.sgd.linear).
     * @param modelVersion A version number for this model.
     * @param outputPath The path to write to.
     * @throws IOException if the file could not be written to.
     */
    default public void saveONNXModel(String domain, long modelVersion, Path outputPath) throws IOException {
        OnnxMl.ModelProto proto = exportONNXModel(domain,modelVersion);
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(outputPath.toFile()))) {
            proto.writeTo(bos);
        }
    }

    /**
     * Serializes the model provenance to a String.
     * @param provenance The provenance to serialize.
     * @return The serialized form of the ModelProvenance.
     */
    default public String serializeProvenance(ModelProvenance provenance) {
        return SERIALIZER.marshalAndSerialize(provenance);
    }

}
