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

package org.tribuo.interop.tensorflow;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.interop.tensorflow.protos.DenseFeatureConverterProto;
import org.tribuo.interop.tensorflow.protos.FeatureConverterProto;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Converts a sparse example into a dense float vector, then wraps it in a {@link TFloat32}.
 */
@ProtoSerializableClass(serializedDataClass = DenseFeatureConverterProto.class, version = DenseFeatureConverter.CURRENT_VERSION)
public class DenseFeatureConverter implements FeatureConverter {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(DenseFeatureConverter.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Feature size beyond which a warning is generated (as TensorFlow requires dense features and large feature spaces are memory hungry).
     */
    public static final int THRESHOLD = 1000000;

    /**
     * Number of times the feature size warning should be printed.
     */
    public static final int WARNING_THRESHOLD = 10;

    private int warningCount = 0;

    @Config(mandatory=true,description="TensorFlow Placeholder Input name.")
    @ProtoSerializableField
    private String inputName;

    /**
     * For OLCUT.
     */
    private DenseFeatureConverter() {}

    /**
     * Builds a DenseFeatureConverter, setting the input name.
     * @param inputName The placeholder input name.
     */
    public DenseFeatureConverter(String inputName) {
        this.inputName = inputName;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static DenseFeatureConverter deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        DenseFeatureConverterProto proto = message.unpack(DenseFeatureConverterProto.class);
        return new DenseFeatureConverter(proto.getInputName());
    }

    @Override
    public FeatureConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    float[] innerTransform(Example<?> example, ImmutableFeatureMap featureIDMap) {
        if ((warningCount < WARNING_THRESHOLD) && (featureIDMap.size() > THRESHOLD)) {
            logger.warning("Large dense example requested, featureIDMap.size() = " + featureIDMap.size() + ", example.size() = " + example.size());
            warningCount++;
        }
        float[] output = new float[featureIDMap.size()];

        for (Feature f : example) {
            int id = featureIDMap.getID(f.getName());
            if (id > -1) {
                output[id] = (float) f.getValue();
            }
        }

        return output;
    }

    private float[] innerTransform(SGDVector vector) {
        if ((warningCount < WARNING_THRESHOLD) && (vector.size() > THRESHOLD)) {
            logger.warning("Large dense example requested, dimension = " + vector.size() + ", numActiveElements = " + vector.numActiveElements());
            warningCount++;
        }
        float[] output = new float[vector.size()];

        if (vector instanceof DenseVector) {
            DenseVector denseVec = (DenseVector) vector;
            for (int i = 0; i < output.length; i++) {
                output[i] = (float) denseVec.get(i);
            }
        } else {
            // must be sparse
            for (VectorTuple f : vector) {
                output[f.index] = (float) f.value;
            }
        }

        return output;
    }

    @Override
    public TensorMap convert(Example<?> example, ImmutableFeatureMap featureIDMap) {
        float[] output = innerTransform(example,featureIDMap);
        return new TensorMap(inputName,TFloat32.tensorOf(Shape.of(1,output.length), DataBuffers.of(output)));
    }

    @Override
    public TensorMap convert(List<? extends Example<?>> examples, ImmutableFeatureMap featureIDMap) {
        TFloat32 output = TFloat32.tensorOf(Shape.of(examples.size(),featureIDMap.size()));

        int i = 0;
        for (Example<?> example : examples) {
            float[] features = innerTransform(example,featureIDMap);
            output.set(NdArrays.vectorOf(features),i);
            i++;
        }

        return new TensorMap(inputName,output);
    }

    @Override
    public TensorMap convert(SGDVector vector) {
        float[] output = innerTransform(vector);
        return new TensorMap(inputName,TFloat32.tensorOf(Shape.of(1,output.length), DataBuffers.of(output)));
    }

    @Override
    public TensorMap convert(List<? extends SGDVector> vectors) {
        TFloat32 output = TFloat32.tensorOf(Shape.of(vectors.size(),vectors.get(0).size()));

        int i = 0;
        for (SGDVector vector : vectors) {
            float[] features = innerTransform(vector);
            output.set(NdArrays.vectorOf(features),i);
            i++;
        }

        return new TensorMap(inputName,output);
    }

    @Override
    public Set<String> inputNamesSet() {
        return Collections.singleton(inputName);
    }

    @Override
    public String toString() {
        return "DenseFeatureConverter(inputName='"+inputName+"')";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FeatureConverter");
    }
}
