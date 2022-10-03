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
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.interop.onnx.protos.ExampleTransformerProto;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;

import java.util.List;
import java.util.logging.Logger;

/**
 * Converts a sparse Tribuo example into a dense float vector, then wraps it in an {@link OnnxTensor}.
 */
@ProtoSerializableClass(version = DenseTransformer.CURRENT_VERSION)
public class DenseTransformer implements ExampleTransformer {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(DenseTransformer.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Feature size beyond which a warning is generated (as ONNX requires dense features and large feature spaces are memory hungry).
     */
    public static final int THRESHOLD = 1000000;

    /**
     * Number of times the feature size warning should be printed.
     */
    public static final int WARNING_THRESHOLD = 10;

    private int warningCount = 0;

    /**
     * Construct a transformer which converts Tribuo sparse vectors into a dense tensor.
     */
    public DenseTransformer() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static DenseTransformer deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new DenseTransformer();
    }

    private float[] innerTransform(SparseVector vector) {
        if ((warningCount < WARNING_THRESHOLD) && (vector.size() > THRESHOLD)) {
            logger.warning("Large dense example requested, dimension = " + vector.size() + ", numActiveElements = " + vector.numActiveElements());
            warningCount++;
        }
        float[] output = new float[vector.size()];

        for (VectorTuple f : vector) {
            output[f.index] = (float) f.value;
        }

        return output;
    }

    @Override
    public OnnxTensor transform(OrtEnvironment env, SparseVector vector) throws OrtException {
        float[][] output = new float[1][];
        output[0] = innerTransform(vector);
        return OnnxTensor.createTensor(env,output);
    }

    @Override
    public OnnxTensor transform(OrtEnvironment env, List<SparseVector> vectors) throws OrtException {
        float[][] output = new float[vectors.size()][];

        int i = 0;
        for (SparseVector vector : vectors) {
            output[i] = innerTransform(vector);
            i++;
        }

        return OnnxTensor.createTensor(env,output);
    }

    @Override
    public String toString() {
        return "DenseTransformer()";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o != null && getClass() == o.getClass();
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public ExampleTransformerProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ExampleTransformer");
    }
}
