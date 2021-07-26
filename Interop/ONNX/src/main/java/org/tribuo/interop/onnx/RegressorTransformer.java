/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Can convert an {@link OnnxValue} into a {@link Prediction} or {@link Regressor}.
 */
public class RegressorTransformer implements OutputTransformer<Regressor> {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs a RegressorTransformer.
     */
    public RegressorTransformer() {}

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
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputTransformer");
    }
}
