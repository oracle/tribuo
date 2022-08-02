/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.fs;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.CategoricalIDInfo;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.RealIDInfo;
import org.tribuo.VariableIDInfo;
import org.tribuo.classification.Label;
import org.tribuo.math.la.DenseVector;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.transformations.BinningTransformation;
import org.tribuo.util.infotheory.InformationTheory;
import org.tribuo.util.infotheory.impl.CachedPair;
import org.tribuo.util.infotheory.impl.CachedTriple;
import org.tribuo.util.infotheory.impl.PairDistribution;
import org.tribuo.util.infotheory.impl.TripleDistribution;

import java.util.HashMap;
import java.util.Map;

/**
 * A {@link FSMatrix} which densifies the dataset.
 */
final class DenseFSMatrix implements FSMatrix {

    private final int[] labels;
    private final int[][] features;
    private final ImmutableFeatureMap fmap;
    private final int numBins;
    private final int numLabels;

    private DenseFSMatrix(int[] labels, int[][] features, ImmutableFeatureMap fmap, int numBins, int numLabels) {
        this.labels = labels;
        this.features = features;
        this.fmap = fmap;
        this.numBins = numBins;
        this.numLabels = numLabels;
    }

    @Override
    public int getNumFeatures() {
        return features.length;
    }

    @Override
    public int getNumSamples() {
        return labels.length;
    }

    @Override
    public ImmutableFeatureMap getFeatureMap() {
        return fmap;
    }

    @Override
    public double mi(int featureIndex) {
        Map<CachedPair<Integer,Integer>, MutableLong> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            CachedPair<Integer, Integer> p = new CachedPair<>(features[featureIndex][i],labels[i]);
            MutableLong l = map.computeIfAbsent(p, k -> new MutableLong());
            l.increment();
        }
        return InformationTheory.mi(PairDistribution.constructFromMap(map,numBins,numLabels));
    }

    @Override
    public double mi(int firstIndex, int secondIndex) {
        Map<CachedPair<Integer,Integer>, MutableLong> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            CachedPair<Integer, Integer> p = new CachedPair<>(features[firstIndex][i],features[secondIndex][i]);
            MutableLong l = map.computeIfAbsent(p, k -> new MutableLong());
            l.increment();
        }
        return InformationTheory.mi(PairDistribution.constructFromMap(map,numBins,numBins));
    }

    @Override
    public double jmi(int featureIndex, int jointIndex) {
        Map<CachedTriple<Integer,Integer,Integer>, MutableLong> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            CachedTriple<Integer, Integer, Integer> p = new CachedTriple<>(features[featureIndex][i],
                    features[jointIndex][i],labels[i]);
            MutableLong l = map.computeIfAbsent(p, k -> new MutableLong());
            l.increment();
        }
        return InformationTheory.jointMI(TripleDistribution.constructFromMap(map));
    }

    @Override
    public double jmi(int firstIndex, int jointIndex, int targetIndex) {
        Map<CachedTriple<Integer,Integer,Integer>, MutableLong> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            CachedTriple<Integer, Integer, Integer> p = new CachedTriple<>(features[firstIndex][i],
                    features[jointIndex][i],features[targetIndex][i]);
            MutableLong l = map.computeIfAbsent(p, k -> new MutableLong());
            l.increment();
        }
        return InformationTheory.jointMI(TripleDistribution.constructFromMap(map));
    }

    @Override
    public double cmi(int featureIndex, int conditionIndex) {
        Map<CachedTriple<Integer,Integer,Integer>, MutableLong> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            CachedTriple<Integer, Integer, Integer> p = new CachedTriple<>(features[featureIndex][i],
                    labels[i],features[conditionIndex][i]);
            MutableLong l = map.computeIfAbsent(p, k -> new MutableLong());
            l.increment();
        }
        return InformationTheory.conditionalMI(TripleDistribution.constructFromMap(map));
    }

    @Override
    public double cmi(int firstIndex, int secondIndex, int conditionIndex) {
        Map<CachedTriple<Integer,Integer,Integer>, MutableLong> map = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            CachedTriple<Integer, Integer, Integer> p = new CachedTriple<>(features[firstIndex][i],
                    features[secondIndex][i],features[conditionIndex][i]);
            MutableLong l = map.computeIfAbsent(p, k -> new MutableLong());
            l.increment();
        }
        return InformationTheory.conditionalMI(TripleDistribution.constructFromMap(map));
    }

    /**
     * Makes equal width bins for each feature and constructs a dense representation of the feature matrix.
     * @param dataset The dataset to convert.
     * @param numBins The number of bins to use.
     * @return A {@code DenseFSMatrix}.
     */
    static DenseFSMatrix equalWidthBins(Dataset<Label> dataset, int numBins) {
        ImmutableFeatureMap fmap = dataset.getFeatureIDMap();
        ImmutableOutputInfo<Label> lmap = dataset.getOutputIDInfo();
        int numFeatures = fmap.size();
        int numExamples = dataset.size();
        int numLabels = dataset.getOutputInfo().size();

        int[][] features = new int[numFeatures][numExamples];
        int[] labels = new int[numExamples];

        Transformer[] transformers = new Transformer[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            VariableIDInfo info = fmap.get(i);
            transformers[i] = makeBinningTransformer(info,numExamples,numBins);
        }

        for (int i = 0; i < numExamples; i++) {
            Example<Label> ex = dataset.getExample(i);
            DenseVector vec = DenseVector.createDenseVector(ex,fmap,false);
            for (int j = 0; j < numFeatures; j++) {
                int bin = (int) transformers[j].transform(vec.get(j));
                features[j][i] = bin;
            }
            labels[i] = lmap.getID(ex.getOutput());
        }

        return new DenseFSMatrix(labels,features,fmap,numBins,numLabels);
    }

    /**
     * Makes an equal width binning transformer by inspecting the variable info to determine the max and min.
     * @param info The variable info to use.
     * @param numExamples The number of examples (to check if an implicit zero should be added).
     * @param numBins The number of bins.
     * @return The binning transformer.
     */
    private static Transformer makeBinningTransformer(VariableIDInfo info, int numExamples, int numBins) {
        int count = info.getCount();
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;

        if (info instanceof CategoricalIDInfo) {
            CategoricalIDInfo catInfo = (CategoricalIDInfo) info;
            double[] values = catInfo.getValues();
            for (int i = 0; i < values.length; i++) {
                double cur = values[i];
                min = Math.min(min,cur);
                max = Math.max(max,cur);
            }
        } else if (info instanceof RealIDInfo) {
            RealIDInfo realInfo = (RealIDInfo) info;
            min = realInfo.getMin();
            max = realInfo.getMax();
        } else {
            throw new IllegalStateException("Unknown variable info subclass " + info.getClass());
        }

        if (numExamples != count) {
            min = Math.min(min,0);
            max = Math.max(max,0);
        }

        double range = Math.abs(max - min);
        double increment = range / numBins;
        double[] bins = new double[numBins];
        double[] values = new double[numBins];

        for (int i = 0; i < bins.length; i++) {
            bins[i] = min + ((i+1) * increment);
            values[i] = i+1;
        }

        return new BinningTransformation.BinningTransformer(BinningTransformation.BinningType.EQUAL_WIDTH,bins,values);
    }
}
