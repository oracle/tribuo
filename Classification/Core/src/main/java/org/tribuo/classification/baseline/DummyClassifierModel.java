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

package org.tribuo.classification.baseline;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.ImmutableLabelInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.baseline.DummyClassifierTrainer.DummyType;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.Util;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;

import static org.tribuo.Trainer.DEFAULT_SEED;

/**
 * A model which performs dummy classifications (e.g. constant output, uniform sampled labels, stratified sampled labels).
 */
public class DummyClassifierModel extends Model<Label> {
    private static final long serialVersionUID = 1L;

    private final DummyType dummyType;

    private final Label constantLabel;

    private final double[] cdf;

    private final Random rng;

    private final long seed;

    DummyClassifierModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo) {
        super("dummy-MOST_FREQUENT-classifier", description, featureIDMap, outputIDInfo, false);
        this.dummyType = DummyType.MOST_FREQUENT;
        this.constantLabel = findMostFrequentLabel(outputIDInfo);
        this.cdf = null;
        this.seed = DEFAULT_SEED;
        this.rng = null;
    }

    DummyClassifierModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo, DummyType dummyType, long seed) {
        super("dummy-"+dummyType+"-classifier", description, featureIDMap, outputIDInfo, false);
        this.dummyType = dummyType;
        this.constantLabel = LabelFactory.UNKNOWN_LABEL;
        this.cdf = dummyType == DummyType.UNIFORM ? generateUniformCDF(outputIDInfo) : generateStratifiedCDF(outputIDInfo);
        this.seed = seed;
        this.rng = new Random(seed);
    }

    DummyClassifierModel(ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> outputIDInfo, Label constantLabel) {
        super("dummy-CONSTANT-classifier", description, featureIDMap, outputIDInfo, false);
        this.dummyType = DummyType.CONSTANT;
        this.constantLabel = constantLabel;
        this.cdf = null;
        this.seed = DEFAULT_SEED;
        this.rng = null;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        switch (dummyType) {
            case CONSTANT:
            case MOST_FREQUENT:
                return new Prediction<>(constantLabel,0,example);
            case UNIFORM:
            case STRATIFIED:
                return new Prediction<>(sampleLabel(cdf,outputIDInfo,rng),0,example);
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        Map<String,List<Pair<String,Double>>> map = new HashMap<>();
        if (n != 0) {
            map.put(Model.ALL_OUTPUTS, Collections.singletonList(new Pair<>(BIAS_FEATURE, 1.0)));
        }
        return map;
    }

    @Override
    public Optional<Excuse<Label>> getExcuse(Example<Label> example) {
        return Optional.of(new Excuse<>(example,predict(example),getTopFeatures(1)));
    }

    @Override
    protected DummyClassifierModel copy(String newName, ModelProvenance newProvenance) {
        switch (dummyType) {
            case CONSTANT:
                return new DummyClassifierModel(newProvenance,featureIDMap,outputIDInfo,constantLabel.copy());
            case MOST_FREQUENT:
                return new DummyClassifierModel(newProvenance,featureIDMap,outputIDInfo);
            case UNIFORM:
            case STRATIFIED:
                return new DummyClassifierModel(newProvenance,featureIDMap,outputIDInfo,dummyType,seed);
            default:
                throw new IllegalStateException("Unknown dummyType " + dummyType);
        }
    }

    /**
     * Samples a label from the supplied CDF.
     * @param cdf The CDF to sample from.
     * @param outputIDInfo The mapping from label ids to values.
     * @param rng The RNG to use.
     * @return A Label.
     */
    private static Label sampleLabel(double[] cdf, ImmutableOutputInfo<Label> outputIDInfo, Random rng) {
        int sample = Util.sampleFromCDF(cdf,rng);
        return outputIDInfo.getOutput(sample);
    }

    /**
     * Finds the most frequent label and returns it.
     * @param outputInfo The output information (must be a subclass of ImmutableLabelInfo).
     * @return The most frequent label.
     */
    private static Label findMostFrequentLabel(ImmutableOutputInfo<Label> outputInfo) {
        Label maxLabel = null;
        long count = -1;

        ImmutableLabelInfo labelInfo = (ImmutableLabelInfo) outputInfo;

        for (Pair<Integer,Label> p : labelInfo) {
            long curCount = labelInfo.getLabelCount(p.getA());
            if (curCount > count) {
                count = curCount;
                maxLabel = p.getB();
            }
        }

        return maxLabel;
    }

    /**
     * Generates a uniform CDF for the supplied labels.
     * @param outputInfo The output information.
     * @return A uniform CDF across the domain.
     */
    private static double[] generateUniformCDF(ImmutableOutputInfo<Label> outputInfo) {
        int length = outputInfo.getDomain().size();
        double[] pmf = Util.generateUniformVector(length,1.0/length);
        return Util.generateCDF(pmf);
    }

    /**
     * Generates a CDF where the label probabilities are proportional to their observed counts.
     * @param outputInfo The output information.
     * @return A CDF proportional to the observed counts.
     */
    private static double[] generateStratifiedCDF(ImmutableOutputInfo<Label> outputInfo) {
        ImmutableLabelInfo labelInfo = (ImmutableLabelInfo) outputInfo;
        int length = labelInfo.getDomain().size();
        long counts = labelInfo.getTotalObservations();

        double[] pmf = new double[length];

        for (Pair<Integer,Label> p : labelInfo) {
            int idx = p.getA();
            pmf[idx] = labelInfo.getLabelCount(idx) / (double) counts;
        }

        return Util.generateCDF(pmf);
    }
}
