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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.regression.Regressor;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Can convert a {@link Regressor} to a {@link Tensor} containing a 32-bit float and a Tensor into a
 * {@link Prediction} or Regressor.
 */
public class RegressorTransformer implements OutputTransformer<Regressor> {
    private static final long serialVersionUID = 1L;

    @Override
    public Prediction<Regressor> transformToPrediction(Tensor<?> tensor, ImmutableOutputInfo<Regressor> outputIDInfo, int numValidFeatures, Example<Regressor> example) {
        Regressor r = transformToOutput(tensor,outputIDInfo);
        return new Prediction<>(r,numValidFeatures,example);
    }

    @Override
    public Regressor transformToOutput(Tensor<?> tensor, ImmutableOutputInfo<Regressor> outputIDInfo) {
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

    float[][] getBatchPredictions(Tensor<?> tensor) {
        long[] shape = tensor.shape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Supplied tensor has the wrong number of dimensions, shape = " + Arrays.toString(shape));
        }
        int numValues = (int) shape[1];
        if (numValues != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many elements, tensor.length = " + numValues);
        }
        int batchSize = (int) shape[0];
        Tensor<Float> converted = tensor.expect(Float.class);
        return converted.copyTo(new float[batchSize][numValues]);
    }

    @Override
    public List<Prediction<Regressor>> transformToBatchPrediction(Tensor<?> tensor, ImmutableOutputInfo<Regressor> outputIDInfo, int[] numValidFeatures, List<Example<Regressor>> examples) {
        List<Regressor> regressors = transformToBatchOutput(tensor,outputIDInfo);
        List<Prediction<Regressor>> output = new ArrayList<>();

        if ((regressors.size() != examples.size()) || (regressors.size() != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from Tensorflow, expected " + numValidFeatures.length + ", received " + regressors.size());
        }

        for (int i = 0; i < regressors.size(); i++) {
            output.add(new Prediction<>(regressors.get(i),numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<Regressor> transformToBatchOutput(Tensor<?> tensor, ImmutableOutputInfo<Regressor> outputIDInfo) {
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
    public Tensor<?> transform(Regressor example, ImmutableOutputInfo<Regressor> outputIDInfo) {
        float[] output = new float[example.size()];
        double[] values = example.getValues();
        for (int i = 0; i < output.length; i++) {
            output[i] = (float) values[i];
        }
        return Tensor.create(output);
    }

    @Override
    public Tensor<?> transform(List<Example<Regressor>> examples, ImmutableOutputInfo<Regressor> outputIDInfo) {
        float[][] output = new float[examples.size()][outputIDInfo.size()];
        int i = 0;
        for (Example<Regressor> e : examples) {
            double[] values = e.getOutput().getValues();
            for (int j = 0; j < output.length; j++) {
                output[i][j] = (float) values[j];
            }
            i++;
        }
        return Tensor.create(output);
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
