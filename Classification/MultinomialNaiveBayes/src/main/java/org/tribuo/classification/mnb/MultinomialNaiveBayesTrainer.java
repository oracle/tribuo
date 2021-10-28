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

package org.tribuo.classification.mnb;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.classification.Label;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;

import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * A {@link Trainer} which trains a multinomial Naive Bayes model with Laplace smoothing.
 * <p>
 * All feature values must be non-negative.
 * <p>
 * See:
 * <pre>
 * Wang S, Manning CD.
 * "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification"
 * Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics, 2012.
 * </pre>
 */
public class MultinomialNaiveBayesTrainer implements Trainer<Label>, WeightedExamples {

    @Config(description="Smoothing parameter.")
    private double alpha = 1.0;

    private int trainInvocationCount = 0;

    /**
     * Constructs a multinomial naive bayes trainer using a smoothing value of 1.0.
     */
    public MultinomialNaiveBayesTrainer() {
        this(1.0);
    }

    /**
     * Constructs a multinomial naive bayes trainer with the specified smoothing value.
     * @param alpha The smoothing value.
     */
    //TODO support different alphas for different features?
    public MultinomialNaiveBayesTrainer(double alpha) {
        if(alpha <= 0.0) {
            throw new IllegalArgumentException("alpha parameter must be > 0");
        }
        this.alpha = alpha;
    }

    @Override
    public Model<Label> train(Dataset<Label> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public Model<Label> train(Dataset<Label> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        ImmutableOutputInfo<Label> labelInfos = examples.getOutputIDInfo();
        ImmutableFeatureMap featureInfos = examples.getFeatureIDMap();

        Map<Integer, Map<Integer, Double>> labelWeights = new HashMap<>();

        for (Pair<Integer,Label> label : labelInfos) {
            labelWeights.put(label.getA(), new HashMap<>());
        }

        for (Example<Label> ex : examples) {
            int idx = labelInfos.getID(ex.getOutput());
            Map<Integer, Double> featureMap = labelWeights.get(idx);
            double curWeight = ex.getWeight();
            for (Feature feat : ex) {
                if (feat.getValue() < 0.0) {
                    throw new IllegalStateException("Multinomial Naive Bayes requires non-negative features. Found feature " + feat.toString());
                }
                featureMap.merge(featureInfos.getID(feat.getName()), curWeight*feat.getValue(), Double::sum);
            }
        }
        if(invocationCount != INCREMENT_INVOCATION_COUNT) {
            setInvocationCount(invocationCount);
        }
        TrainerProvenance trainerProvenance = getProvenance();
        ModelProvenance provenance = new ModelProvenance(MultinomialNaiveBayesModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        trainInvocationCount++;

        SparseVector[] labelVectors = new SparseVector[labelInfos.size()];

        for(int i = 0; i < labelInfos.size(); i++) {
            SparseVector sv = SparseVector.createSparseVector(featureInfos.size(), labelWeights.get(i));
            double unsmoothedZ = sv.oneNorm();
            sv.foreachInPlace(d -> Math.log((d + alpha) / (unsmoothedZ + (featureInfos.size() * alpha))));
            labelVectors[i] = sv;
        }

        DenseSparseMatrix labelWordProbs = DenseSparseMatrix.createFromSparseVectors(labelVectors);

        return new MultinomialNaiveBayesModel("", provenance, featureInfos, labelInfos, labelWordProbs, alpha);
    }

    @Override
    public int getInvocationCount() {
        return trainInvocationCount;
    }

    @Override
    public void setInvocationCount(int invocationCount) {
        if(invocationCount < 0){
            throw new IllegalArgumentException("The supplied invocationCount is less than zero.");
        }

        this.trainInvocationCount = invocationCount;
    }

    @Override
    public String toString() {
        return "MultinomialNaiveBayesTrainer(alpha=" + alpha + ")";
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
