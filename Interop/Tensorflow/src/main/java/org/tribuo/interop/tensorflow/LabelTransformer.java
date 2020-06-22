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
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tensorflow.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Can convert a {@link Label} into a {@link Tensor} containing a 32-bit integer and
 * can convert a vector of 32-bit floats into a {@link Prediction} or a {@link Label}.
 */
public class LabelTransformer implements OutputTransformer<Label> {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LabelTransformer.class.getName());

    @Override
    public Prediction<Label> transformToPrediction(Tensor<?> tensor, ImmutableOutputInfo<Label> outputIDInfo, int numValidFeatures, Example<Label> example) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        }
        return generatePrediction(predictions[0],outputIDInfo,numValidFeatures,example);
    }

    private Prediction<Label> generatePrediction(float[] predictions, ImmutableOutputInfo<Label> outputIDInfo, int numUsed, Example<Label> example) {
        Label max = null;
        Map<String,Label> map = new HashMap<>();
        for (int i = 0; i < predictions.length; i++) {
            Label current = new Label(outputIDInfo.getOutput(i).getLabel(),predictions[i]);
            map.put(current.getLabel(),current);
            if ((max == null) || (current.getScore() > max.getScore())) {
                max = current;
            }
        }
        return new Prediction<>(max,map,numUsed,example,true);
    }

    @Override
    public Label transformToOutput(Tensor<?> tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        }
        return generateLabel(predictions[0],outputIDInfo);
    }

    private Label generateLabel(float[] predictions, ImmutableOutputInfo<Label> outputIDInfo) {
        int maxIdx = 0;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i] > max) {
                maxIdx = i;
                max = predictions[i];
            }
        }
        return new Label(outputIDInfo.getOutput(maxIdx).getLabel(),max);
    }

    private float[][] getBatchPredictions(Tensor<?> tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        long[] shape = tensor.shape();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Supplied tensor has the wrong number of dimensions, shape = " + Arrays.toString(shape));
        }
        int numValues = (int) shape[1];
        if (numValues != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has too many elements, tensor.length = " + numValues + ", outputIDInfo.size() = " + outputIDInfo.size());
        }
        int batchSize = (int) shape[0];
        Tensor<Float> converted = tensor.expect(Float.class);
        return converted.copyTo(new float[batchSize][numValues]);
    }

    @Override
    public List<Prediction<Label>> transformToBatchPrediction(Tensor<?> tensor, ImmutableOutputInfo<Label> outputIDInfo, int[] numValidFeatures, List<Example<Label>> examples) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Prediction<Label>> output = new ArrayList<>();

        if ((predictions.length != examples.size()) || (predictions.length != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from Tensorflow, expected " + numValidFeatures.length + ", received " + predictions.length);
        }

        for (int i = 0; i < predictions.length; i++) {
            output.add(generatePrediction(predictions[i],outputIDInfo,numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<Label> transformToBatchOutput(Tensor<?> tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        float[][] predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Label> output = new ArrayList<>();

        for (int i = 0; i < predictions.length; i++) {
            output.add(generateLabel(predictions[i],outputIDInfo));
        }

        return output;
    }

    private int innerTransform(Label label, ImmutableOutputInfo<Label> outputIDInfo) {
        int id = outputIDInfo.getID(label);
        if (id == -1) {
            throw new IllegalArgumentException("Label " + label + " isn't known by the supplied outputIDInfo, " + outputIDInfo.toString());
        }
        return id;
    }

    @Override
    public Tensor<?> transform(Label example, ImmutableOutputInfo<Label> outputIDInfo) {
        int[] output = new int[1];
        output[0] = innerTransform(example, outputIDInfo);
        return Tensor.create(output);
    }

    @Override
    public Tensor<?> transform(List<Example<Label>> examples, ImmutableOutputInfo<Label> outputIDInfo) {
        int[] output = new int[examples.size()];
        int i = 0;
        for (Example<Label> e : examples) {
            output[i] = innerTransform(e.getOutput(), outputIDInfo);
            i++;
        }
        return Tensor.create(output);
    }

    @Override
    public boolean generatesProbabilities() {
        return true;
    }

    @Override
    public String toString() {
        return "LabelTransformer()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputTransformer");
    }
}
