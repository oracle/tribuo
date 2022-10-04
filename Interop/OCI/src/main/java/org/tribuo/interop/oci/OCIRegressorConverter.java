/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.oci;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.interop.oci.protos.OCIOutputConverterProto;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.List;

/**
 * A converter for {@link DenseMatrix} and {@link DenseVector} into {@link Regressor} {@link Prediction}s.
 */
@ProtoSerializableClass(version = OCIRegressorConverter.CURRENT_VERSION)
public final class OCIRegressorConverter implements OCIOutputConverter<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs an OCIRegressorConverter.
     */
    public OCIRegressorConverter() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static OCIRegressorConverter deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new OCIRegressorConverter();
    }

    @Override
    public Prediction<Regressor> convertOutput(DenseVector scores, int numValidFeature, Example<Regressor> example, ImmutableOutputInfo<Regressor> outputIDInfo) {
        if (scores.size() != outputIDInfo.size()) {
            throw new IllegalStateException("Expected scores for each output, received " + scores.size() + " when there are " + outputIDInfo.size() + "outputs");
        } else {
            // Note this inserts in an ordering which is not necessarily the natural one,
            // but the Regressor constructor sorts it to maintain the natural ordering.
            // The names and the values still line up, so this code is valid.
            String[] names = new String[outputIDInfo.size()];
            double[] values = new double[outputIDInfo.size()];
            for (Pair<Integer,Regressor> p : outputIDInfo) {
                int id = p.getA();
                names[id] = p.getB().getNames()[0];
                values[id] = scores.get(id);
            }
            return new Prediction<>(new Regressor(names,values),numValidFeature,example);
        }
    }

    @Override
    public List<Prediction<Regressor>> convertOutput(DenseMatrix scores, int[] numValidFeatures, List<Example<Regressor>> examples, ImmutableOutputInfo<Regressor> outputIDInfo) {
        if (scores.getDimension1Size() != examples.size()) {
            throw new IllegalStateException("Expected one prediction per example, recieved " + scores.getDimension1Size() + " predictions when there are " + examples.size() + " examples.");
        }
        List<Prediction<Regressor>> predictions = new ArrayList<>();
        if (scores.getDimension2Size() != outputIDInfo.size()) {
            throw new IllegalStateException("Expected scores for each output, received " + scores.getDimension2Size() + " when there are " + outputIDInfo.size() + "outputs");
        } else {
            // Similar to convertOutput(DenseVector), names and values are ordered by
            // the id, not the natural ordering, but the Regressor constructor
            // fixes that.
            String[] names = new String[outputIDInfo.size()];
            for (Pair<Integer,Regressor> p : outputIDInfo) {
                int id = p.getA();
                names[id] = p.getB().getNames()[0];
            }
            for (int i = 0; i < scores.getDimension1Size(); i++) {
                double[] values = new double[names.length];
                for (int j = 0; j < names.length; j++) {
                    values[j] = scores.get(i,j);
                }
                predictions.add(new Prediction<>(new Regressor(names,values),numValidFeatures[i],examples.get(i)));
            }
        }
        return predictions;
    }

    @Override
    public boolean generatesProbabilities() {
        return false;
    }

    @Override
    public Class<Regressor> getTypeWitness() {
        return Regressor.class;
    }

    @Override
    public String toString() {
        return "OCIRegressorConverter()";
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
    public OCIOutputConverterProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OCIOutputConverter");
    }
}
