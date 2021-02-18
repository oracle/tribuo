/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.index.Indices;
import org.tensorflow.types.TFloat16;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
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
 * can convert a {@link TFloat16} or {@link TFloat32} into a {@link Prediction} or a {@link Label}.
 */
public class LabelTransformer implements OutputTransformer<Label> {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(LabelTransformer.class.getName());

    /**
     * Constructs a LabelTransformer.
     */
    public LabelTransformer() {}

    @Override
    public Prediction<Label> transformToPrediction(Tensor tensor, ImmutableOutputInfo<Label> outputIDInfo, int numValidFeatures, Example<Label> example) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        long batchSize = predictions.shape().asArray()[0];
        if (batchSize != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, batchSize = " + batchSize);
        }
        return generatePrediction(predictions.slice(Indices.at(0),Indices.all()),outputIDInfo,numValidFeatures,example);
    }

    private Prediction<Label> generatePrediction(FloatNdArray predictions, ImmutableOutputInfo<Label> outputIDInfo, int numUsed, Example<Label> example) {
        long[] shape = predictions.shape().asArray();
        if (shape.length != 1) {
            throw new IllegalArgumentException("Failed to get scalar predictions. Found " + Arrays.toString(shape));
        }
        if (shape[0] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("More than Integer.MAX_VALUE predictions. Found " + shape[0]);
        }
        int length = (int) shape[0];
        Label max = null;
        Map<String,Label> map = new HashMap<>();
        for (int i = 0; i < length; i++) {
            Label current = new Label(outputIDInfo.getOutput(i).getLabel(),predictions.getFloat(i));
            map.put(current.getLabel(),current);
            if ((max == null) || (current.getScore() > max.getScore())) {
                max = current;
            }
        }
        return new Prediction<>(max,map,numUsed,example,true);
    }

    @Override
    public Label transformToOutput(Tensor tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        long batchSize = predictions.shape().asArray()[0];
        if (batchSize != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, batchSize = " + batchSize);
        }
        return generateLabel(predictions.slice(Indices.at(0),Indices.all()),outputIDInfo);
    }

    private Label generateLabel(FloatNdArray predictions, ImmutableOutputInfo<Label> outputIDInfo) {
        long[] shape = predictions.shape().asArray();
        if (shape.length != 1) {
            throw new IllegalArgumentException("Failed to get scalar predictions. Found " + Arrays.toString(shape));
        }
        if (shape[0] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("More than Integer.MAX_VALUE predictions. Found " + shape[0]);
        }
        int length = (int) shape[0];
        int maxIdx = 0;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            float pred = predictions.getFloat(i);
            if (pred > max) {
                maxIdx = i;
                max = pred;
            }
        }
        return new Label(outputIDInfo.getOutput(maxIdx).getLabel(),max);
    }

    private FloatNdArray getBatchPredictions(Tensor tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        long[] shape = tensor.shape().asArray();
        if (shape.length != 2) {
            throw new IllegalArgumentException("Supplied tensor has the wrong number of dimensions, shape = " + Arrays.toString(shape));
        }
        int numValues = (int) shape[1];
        if (numValues != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has too many elements, tensor.length = " + numValues + ", outputIDInfo.size() = " + outputIDInfo.size());
        }
        if (tensor instanceof TFloat16) {
            return (TFloat16) tensor;
        } else if (tensor instanceof TFloat32) {
            return (TFloat32) tensor;
        } else {
            throw new IllegalArgumentException("Tensor is not a probability distribution. Found type " + tensor.getClass().getName());
        }
    }

    @Override
    public List<Prediction<Label>> transformToBatchPrediction(Tensor tensor, ImmutableOutputInfo<Label> outputIDInfo, int[] numValidFeatures, List<Example<Label>> examples) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Prediction<Label>> output = new ArrayList<>();

        int batchSize = (int) predictions.shape().asArray()[0];

        if ((batchSize != examples.size()) || (batchSize != numValidFeatures.length)) {
            throw new IllegalArgumentException("Invalid number of predictions received from Tensorflow, expected " + numValidFeatures.length + ", received " + batchSize);
        }

        for (int i = 0; i < batchSize; i++) {
            FloatNdArray slice = predictions.slice(Indices.at(i),Indices.all());
            output.add(generatePrediction(slice,outputIDInfo,numValidFeatures[i],examples.get(i)));
        }

        return output;
    }

    @Override
    public List<Label> transformToBatchOutput(Tensor tensor, ImmutableOutputInfo<Label> outputIDInfo) {
        FloatNdArray predictions = getBatchPredictions(tensor,outputIDInfo);
        List<Label> output = new ArrayList<>();

        int batchSize = (int) predictions.shape().asArray()[0];

        for (int i = 0; i < batchSize; i++) {
            FloatNdArray slice = predictions.slice(Indices.at(i),Indices.all());
            output.add(generateLabel(slice,outputIDInfo));
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
    public Tensor transform(Label example, ImmutableOutputInfo<Label> outputIDInfo) {
        int[] output = new int[1];
        output[0] = innerTransform(example, outputIDInfo);
        return TInt32.vectorOf(output);
    }

    @Override
    public Tensor transform(List<Example<Label>> examples, ImmutableOutputInfo<Label> outputIDInfo) {
        int[] output = new int[examples.size()];
        int i = 0;
        for (Example<Label> e : examples) {
            output[i] = innerTransform(e.getOutput(), outputIDInfo);
            i++;
        }
        return TInt32.vectorOf(output);
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
