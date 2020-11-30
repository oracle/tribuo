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

package org.tribuo.multilabel.sgd.linear;

import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.provenance.ModelProvenance;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The inference time version of a multi-label linear model trained using SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<MultiLabel> {
    private static final long serialVersionUID = 2L;

    private final VectorNormalizer normalizer;
    private final double threshold;

    LinearSGDModel(String name, ModelProvenance description,
                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<MultiLabel> labelIDMap,
                   LinearParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities, double threshold) {
        super(name, description, featureIDMap, labelIDMap, parameters.getWeightMatrix(), generatesProbabilities);
        this.normalizer = normalizer;
        this.threshold = threshold;
    }

    private LinearSGDModel(String name, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<MultiLabel> labelIDMap,
                          DenseMatrix weights, VectorNormalizer normalizer, boolean generatesProbabilities, double threshold) {
        super(name, description, featureIDMap, labelIDMap, weights, generatesProbabilities);
        this.normalizer = normalizer;
        this.threshold = threshold;
    }

    @Override
    public Prediction<MultiLabel> predict(Example<MultiLabel> example) {
        PredAndActive predTuple = predictSingle(example);
        DenseVector outputs = predTuple.prediction;
        outputs.normalize(normalizer);
        Map<String,MultiLabel> fullLabels = new HashMap<>();
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < outputs.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabelString();
            double labelScore = outputs.get(i);
            Label score = new Label(outputIDInfo.getOutput(i).getLabelString(),labelScore);
            if (labelScore > threshold) {
                predictedLabels.add(score);
            }
            fullLabels.put(labelName,new MultiLabel(score));
        }
        return new Prediction<>(new MultiLabel(predictedLabels), fullLabels, predTuple.numActiveFeatures - 1, example, generatesProbabilities);
    }

    @Override
    protected String getDimensionName(int index) {
        return outputIDInfo.getOutput(index).getLabelString();
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,newProvenance,featureIDMap,outputIDInfo,new DenseMatrix(weights),normalizer,generatesProbabilities,threshold);
    }
}
