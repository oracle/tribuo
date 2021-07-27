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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.multilabel.MultiLabel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Can convert an {@link OnnxValue} into a {@link Prediction} or a {@link MultiLabel}.
 * <p>
 * Accepts a single tensor representing the probabilities of each label in the batch.
 * <p>
 * Predictions are thresholded at {@link #THRESHOLD}, probabilities above this are considered to be present in the
 * output.
 */
public class MultiLabelTransformer implements OutputTransformer<MultiLabel> {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(MultiLabelTransformer.class.getName());

    /**
     * The threshold for conversion into a label.
     */
    public static final double THRESHOLD = 0.5;

    /**
     * Constructs a MultiLabelTransformer.
     */
    public MultiLabelTransformer() {}

    @Override
    public Prediction<MultiLabel> transformToPrediction(List<OnnxValue> value, ImmutableOutputInfo<MultiLabel> outputIDInfo, int numValidFeatures, Example<MultiLabel> example) {
        float[][] predictions = getBatchPredictions(value);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        }
        return getPrediction(predictions[0],outputIDInfo,numValidFeatures,example);
    }

    @Override
    public MultiLabel transformToOutput(List<OnnxValue> value, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        float[][] predictions = getBatchPredictions(value);
        if (predictions.length != 1) {
            throw new IllegalArgumentException("Supplied tensor has too many results, predictions.length = " + predictions.length);
        }
        return getOutput(predictions[0],outputIDInfo);
    }

    @Override
    public List<Prediction<MultiLabel>> transformToBatchPrediction(List<OnnxValue> value, ImmutableOutputInfo<MultiLabel> outputIDInfo, int[] numValidFeatures, List<Example<MultiLabel>> examples) {
        float[][] predictions = getBatchPredictions(value);
        if (predictions.length != examples.size()) {
            throw new IllegalArgumentException("Supplied tensor has the wrong number of results, predictions.length = " + predictions.length + ", examples.size() = " + examples.size());
        }
        List<Prediction<MultiLabel>> outputs = new ArrayList<>();
        for (int i = 0; i < predictions.length; i++) {
            outputs.add(getPrediction(predictions[i],outputIDInfo,numValidFeatures[i],examples.get(i)));
        }
        return outputs;
    }

    @Override
    public List<MultiLabel> transformToBatchOutput(List<OnnxValue> value, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        float[][] predictions = getBatchPredictions(value);
        List<MultiLabel> outputs = new ArrayList<>();
        for (float[] curPredictions : predictions) {
            outputs.add(getOutput(curPredictions,outputIDInfo));
        }
        return outputs;
    }

    private MultiLabel getOutput(float[] predictions, ImmutableOutputInfo<MultiLabel> outputIDInfo) {
        if (predictions.length != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has an incorrect number of dimensions, predictions[0].length = " + predictions.length + ", expected " + outputIDInfo.size());
        }
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < predictions.length; i++) {
            double labelScore = predictions[i];
            if (labelScore > THRESHOLD) {
                Label score = new Label(outputIDInfo.getOutput(i).getLabelString(),labelScore);
                predictedLabels.add(score);
            }
        }
        return new MultiLabel(predictedLabels);
    }

    private Prediction<MultiLabel> getPrediction(float[] predictions, ImmutableOutputInfo<MultiLabel> outputIDInfo, int numValidFeatures, Example<MultiLabel> example) {
        if (predictions.length != outputIDInfo.size()) {
            throw new IllegalArgumentException("Supplied tensor has an incorrect number of dimensions, predictions[0].length = " + predictions.length + ", expected " + outputIDInfo.size());
        }
        Map<String,MultiLabel> fullLabels = new HashMap<>(outputIDInfo.size());
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < predictions.length; i++) {
            double labelScore = predictions[i];
            String labelName = outputIDInfo.getOutput(i).getLabelString();
            Label score = new Label(labelName,labelScore);
            if (labelScore > THRESHOLD) {
                predictedLabels.add(score);
            }
            fullLabels.put(labelName,new MultiLabel(score));
        }
        return new Prediction<>(new MultiLabel(predictedLabels), fullLabels, numValidFeatures, example, true);
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
                throw new IllegalArgumentException("Expected shape [batchSize,numLabels], found " + Arrays.toString(shape));
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
    public boolean generatesProbabilities() {
        return true;
    }

    @Override
    public String toString() {
        return "MultiLabelTransformer()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"OutputTransformer");
    }
}
