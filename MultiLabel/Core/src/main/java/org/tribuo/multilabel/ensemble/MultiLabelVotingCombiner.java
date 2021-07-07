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

package org.tribuo.multilabel.ensemble;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.ensemble.EnsembleCombiner;
import org.tribuo.math.la.DenseVector;
import org.tribuo.multilabel.MultiLabel;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A combiner which performs a weighted or unweighted vote independently across the predicted labels in each multi-label.
 * <p>
 * This uses the thresholded predictions from each ensemble member.
 * <p>
 * This class is stateless and thread safe.
 */
public final class MultiLabelVotingCombiner implements EnsembleCombiner<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs a voting combiner.
     */
    public MultiLabelVotingCombiner() {}

    @Override
    public Prediction<MultiLabel> combine(ImmutableOutputInfo<MultiLabel> outputInfo, List<Prediction<MultiLabel>> predictions) {
        int numPredictions = predictions.size();
        double weight = 1.0 / numPredictions;
        int numUsed = 0;
        double[] posScore = new double[outputInfo.size()];
        double[] negScore = new double[outputInfo.size()];
        for (Prediction<MultiLabel> p : predictions) {
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            DenseVector v = p.getOutput().convertToDenseVector(outputInfo);
            for (int j = 0; j < v.size(); j++) {
                double score = v.get(j);
                if (score > 0.5) {
                    posScore[j] += weight;
                } else {
                    negScore[j] += weight;
                }
            }
        }

        Map<String,MultiLabel> fullLabels = new LinkedHashMap<>();
        Set<Label> predSet = new HashSet<>();
        for (int i = 0; i < posScore.length; i++) {
            String name = outputInfo.getOutput(i).getLabelString();
            double score = posScore[i] / (posScore[i] + negScore[i]);
            Label label = new Label(name, score);
            if (score > 0.5) {
                predSet.add(label);
            }
            fullLabels.put(name,new MultiLabel(label));
        }

        Example<MultiLabel> example = predictions.get(0).getExample();

        return new Prediction<>(new MultiLabel(predSet),fullLabels,numUsed,example,true);
    }

    @Override
    public Prediction<MultiLabel> combine(ImmutableOutputInfo<MultiLabel> outputInfo, List<Prediction<MultiLabel>> predictions, float[] weights) {
        if (predictions.size() != weights.length) {
            throw new IllegalArgumentException("predictions and weights must be the same length. predictions.size()="+predictions.size()+", weights.length="+weights.length);
        }
        int numUsed = 0;
        double[] posScore = new double[outputInfo.size()];
        double[] negScore = new double[outputInfo.size()];
        for (int i = 0; i < weights.length; i++) {
            Prediction<MultiLabel> p = predictions.get(i);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            DenseVector v = p.getOutput().convertToDenseVector(outputInfo);
            for (int j = 0; j < v.size(); j++) {
               double score = v.get(j);
               if (score > 0.5) {
                   posScore[j] += weights[i];
               } else {
                   negScore[j] += weights[i];
               }
            }
        }

        Map<String,MultiLabel> fullLabels = new LinkedHashMap<>();
        Set<Label> predSet = new HashSet<>();
        for (int i = 0; i < posScore.length; i++) {
            String name = outputInfo.getOutput(i).getLabelString();
            double score = posScore[i] / (posScore[i] + negScore[i]);
            Label label = new Label(name, score);
            if (score > 0.5) {
                predSet.add(label);
            }
            fullLabels.put(name,new MultiLabel(label));
        }

        Example<MultiLabel> example = predictions.get(0).getExample();

        return new Prediction<>(new MultiLabel(predSet),fullLabels,numUsed,example,true);
    }

    @Override
    public String toString() {
        return "MultiLabelVotingCombiner()";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"EnsembleCombiner");
    }
}
