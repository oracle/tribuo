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
     * Exports this {@link org.tribuo.Model} as an ONNX graph proto.
     * <p>
     * This graph can be combined with other graphs to form an ensemble or other
     * aggregate ONNX model.
     * @param context The ONNX context to use for namespacing.
     * @return The ONNX GraphProto representing this Tribuo Model.
     */
    public OnnxMl.GraphProto exportONNXGraph(ONNXContext context);

    /**
     * Exports this {@link org.tribuo.Model} as an ONNX graph proto.
     * <p>
     * This graph can be combined with other graphs to form an ensemble or other
     * aggregate ONNX model.
     * <p>
     * Creates a fresh ONNX context.
     * @return The ONNX GraphProto representing this Tribuo Model.
     */
    default public OnnxMl.GraphProto exportONNXGraph() {
        return exportONNXGraph(new ONNXContext());
    }

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
