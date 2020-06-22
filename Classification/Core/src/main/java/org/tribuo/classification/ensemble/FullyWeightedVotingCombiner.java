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

package org.tribuo.classification.ensemble;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.ensemble.EnsembleCombiner;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A combiner which performs a weighted or unweighted vote across the predicted labels.
 * <p>
 * This uses the full distribution of predictions from each ensemble member, unlike {@link VotingCombiner}
 * which uses the most likely prediction for each ensemble member.
 */
public final class FullyWeightedVotingCombiner implements EnsembleCombiner<Label> {
    private static final long serialVersionUID = 1L;

    public FullyWeightedVotingCombiner() {}

    @Override
    public Prediction<Label> combine(ImmutableOutputInfo<Label> outputInfo, List<Prediction<Label>> predictions) {
        int numPredictions = predictions.size();
        int numUsed = 0;
        double weight = 1.0 / numPredictions;
        double sum = 0.0;
        double[] score = new double[outputInfo.size()];
        for (Prediction<Label> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            for (Label e : p.getOutputScores().values()) {
                double curScore = weight * e.getScore();
                sum += curScore;
                score[outputInfo.getID(e)] += curScore;
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predictionMap = new LinkedHashMap<>();
        for (int i = 0; i < score.length; i++) {
            String name = outputInfo.getOutput(i).getLabel();
            Label label = new Label(name,score[i]/sum);
            predictionMap.put(name,label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        Example<Label> example = predictions.get(0).getExample();

        return new Prediction<>(maxLabel,predictionMap,numUsed,example,true);
    }

    @Override
    public Prediction<Label> combine(ImmutableOutputInfo<Label> outputInfo, List<Prediction<Label>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()="+predictions.size()+", weights.length="+weights.length);
        }
        int numUsed = 0;
        double sum = 0.0;
        double[] score = new double[outputInfo.size()];
        for (int i = 0; i < weights.length; i++) {
            Prediction<Label> p = predictions.get(i);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            for (Label e : p.getOutputScores().values()) {
                double curScore = weights[i] * e.getScore();
                sum += curScore;
                score[outputInfo.getID(e)] += curScore;
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predictionMap = new LinkedHashMap<>();
        for (int i = 0; i < score.length; i++) {
            String name = outputInfo.getOutput(i).getLabel();
            Label label = new Label(name,score[i]/sum);
            predictionMap.put(name,label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }

        Example<Label> example = predictions.get(0).getExample();

        return new Prediction<>(maxLabel,predictionMap,numUsed,example,true);
    }

    @Override
    public String toString() {
        return "FullyWeightedVotingCombiner()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"EnsembleCombiner");
    }
}
