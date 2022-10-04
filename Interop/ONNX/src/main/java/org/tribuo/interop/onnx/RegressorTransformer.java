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

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.interop.onnx.protos.OutputTransformerProto;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Can convert an {@link OnnxValue} into a {@link Prediction} or {@link Regressor}.
 */
@ProtoSerializableClass(version = RegressorTransformer.CURRENT_VERSION)
public class RegressorTransformer implements OutputTransformer<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a RegressorTransformer.
     */
    public RegressorTransformer() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static RegressorTransformer deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new RegressorTransformer();
    }

    @Override
    public Prediction<Regressor> transformToPrediction(List<OnnxValue> tensor, ImmutableOutputInfo<Regressor> outputIDInfo, int numValidFeatures, Example<Regressor> example) {
        Regressor r = transformToOutput(tensor,outputIDInfo);
        return new Prediction<>(r,numValidFeatures,example);
    }

    @Override
    public Regressor transformToOutput(List<OnnxValue> tensor, ImmutableOutputInfo<Regressor> outputIDInfo) {
        float[][] predictions = getBatchPredictions(tensor);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        } else if (predictions[0].length != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has an incorrect number of dimensions, predictions[0].length = " + predictions[0].length + ", expected " + outputIDInfo.size());
        }
        // Note this inserts in an ordering which is not necessarily the natural one,
        // but the Regressor constructor sorts it to maintain the natural ordering.
        // The names and the values still line up, so this code is valid.
        String[] names = new String[outputIDInfo.size()];
        double[] values = new double[outputIDInfo.size()];
        for (Pair<Integer,Regressor> p : outputIDInfo) {
            int id = p.getA();
            names[id] = p.getB().getNames()[0];
            values[id] = predictions[0][id];
        }
        return new Regressor(names,values);
    }

    private float[][] getBatchPredictions(List<OnnxValue> valueList) {
        if (valueList.size() != 1) {
            throw new IllegalArgumentException("Supplied output has incorrect number of elements, expected 1, found " + valueList.size());
        }

        OnnxValue value = valueList.get(0);
        if (value instanceof OnnxTensor) {
            OnnxTensor tensor = (OnnxTensor) value;
            long[] shape = tensor.getInfo().getShape();
            if (shape.length != 2) {
                throw new IllegalArgumentException("Expected shape [batchSize][numDimensions], found " + Arrays.toString(shape));
            } else {
                try {
                    if (tensor.getInfo().type == OnnxJavaType.FLOAT) {
                        // Will return a float array
                        return (float[][]) tensor.getValue();
                    } else {
                        throw new IllegalArgumentException("Supplied output was an invalid tensor type, expected float, found " + tensor.getInfo().type);
                    }
                } catch (OrtException e) {
                    throw new IllegalStateException("Failed to read tensor value",e);
                }
            }
        } else {
            throw new IllegalArgumentException("Supplied output was not an OnnxTensor, found " + value.getClass().toString());
        }
    }

    @Override
    public List<Prediction<Regressor>> transformToBatchPrediction(List<OnnxValue> tensor, ImmutableOutputInfo<Regressor> outputIDInfo, int[] numValidFeatures, List<Example<Regressor>> examples) {
        List<Regressor> regressors = transformToBatchOutput(tensor,outputIDInfo);
        List<Prediction<Regressor>> output = new ArrayList<>();

        if ((regressors.size() != examples.size()) || (regressors.size() != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from the ONNXExternalModel, expected " + numValidFeatures.length + ", received " + regressors.size());
        }

        for (int i = 0; i < regressors.size(); i++) {
            output.add(new Prediction<>(regressors.get(i),numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<Regressor> transformToBatchOutput(List<OnnxValue> tensor, ImmutableOutputInfo<Regressor> outputIDInfo) {
        float[][] predictions = getBatchPredictions(tensor);
        List<Regressor> output = new ArrayList<>();

        // Similar to transformToOutput, names and values are ordered by
        // the id, not the natural ordering, but the Regressor constructor
        // fixes that.
        String[] names = new String[outputIDInfo.size()];
        for (Pair<Integer,Regressor> p : outputIDInfo) {
            int id = p.getA();
            names[id] = p.getB().getNames()[0];
        }
        for (int i = 0; i < predictions.length; i++) {
            double[] values = new double[names.length];
            for (int j = 0; j < names.length; j++) {
                values[j] = predictions[i][j];
            }
            output.add(new Regressor(names,values));
        }

        return output;
    }

    @Override
    public boolean generatesProbabilities() {
        return false;
    }

    @Override
    public String toString() {
        return "RegressorTransformer()";
    }

    @Override
    public Class<Regressor> getTypeWitness() {
        return Regressor.class;
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
    public OutputTransformerProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputTransformer");
    }
}
