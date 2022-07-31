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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.Dataset;
import org.tribuo.FeatureSelector;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.classification.Label;
import org.tribuo.provenance.FeatureSelectorProvenance;
import org.tribuo.provenance.FeatureSetProvenance;
import org.tribuo.provenance.impl.FeatureSelectorProvenanceImpl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

/**
 * Selects features according to the Conditional Mutual Information Maximisation algorithm.
 * <p>
 * Uses equal width binning for the feature values.
 * <p>
 * See:
 * <pre>
 * Fleuret F.
 * "Fast Binary Feature Selection with Conditional Mutual Information"
 * Journal of Machine Learning Research (JMLR), 2004, <a href="https://www.jmlr.org/papers/volume5/fleuret04a/fleuret04a.pdf">PDF</a>.
 * </pre>
 */
public final class CMIM implements FeatureSelector<Label> {
    private static final Logger logger = Logger.getLogger(CMIM.class.getName());

    @Config(mandatory = true, description = "Number of bins to use when discretising continuous features.")
    private int numBins;

    @Config(description = "Number of features to select, defaults to ranking all features.")
    private int k = SELECT_ALL;

    @Config(description = "Number of computation threads to use.")
    private int numThreads = 1;

    /**
     * For OLCUT.
     */
    private CMIM() { }

    /**
     * Constructs a CMIM feature selector that ranks the top {@code k} features.
     * <p>
     * Continuous features are binned into {@code numBins} equal width bins.
     * @param numBins The number of bins, must be greater than 1.
     * @param k The number of features to rank.
     * @param numThreads The number of computation threads to use.
     */
    public CMIM(int k, int numBins, int numThreads) {
        this.k = k;
        this.numBins = numBins;
        this.numThreads = numThreads;
        if ((k != SELECT_ALL) && (k < 1)) {
            throw new IllegalArgumentException("k must be -1 to select all features, or a positive number, found " + k);
        }
        if (numBins < 2) {
            throw new IllegalArgumentException("numBins must be >= 2, found " + numBins);
        }
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if ((k != SELECT_ALL) && (k < 1)) {
            throw new PropertyException("","k","k must be -1 to select all features, or a positive number, found " + k);
        }
        if (numBins < 2) {
            throw new PropertyException("","numBins","numBins must be >= 2, found " + numBins);
        }
    }

    @Override
    public boolean isOrdered() {
        return true;
    }

    @Override
    public SelectedFeatureSet select(Dataset<Label> dataset) {
        FSMatrix data = FSMatrix.buildMatrix(dataset,numBins);
        ImmutableFeatureMap fmap = data.getFeatureMap();
        int max = k == -1 ? fmap.size() : Math.min(k,fmap.size());
        int numFeatures = fmap.size();

        boolean[] unselectedFeatures = new boolean[numFeatures];
        Arrays.fill(unselectedFeatures, true);
        int[] selectedFeatures = new int[max];
        double[] selectedScores = new double[max];

        double[] miCache;
        int[] idxCache = new int[numFeatures];

        if (numThreads > 1) {
            ForkJoinPool fjp = new ForkJoinPool(numThreads);
            try {
                miCache = fjp.submit(() -> IntStream.range(0, numFeatures).parallel().mapToDouble(data::mi).toArray()).get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
            fjp.shutdown();
        } else {
            miCache = IntStream.range(0, numFeatures).mapToDouble(data::mi).toArray();
        }

        int curIdx = -1;
        double curVal = -1.0;
        for (int i = 0; i < numFeatures; i++) {
            if (miCache[i] > curVal) {
                curIdx = i;
                curVal = miCache[i];
            }
        }

        selectedFeatures[0] = curIdx;
        selectedScores[0] = curVal;
        unselectedFeatures[curIdx] = false;
        logger.log(Level.INFO,"Itr 0: selected feature " + fmap.get(curIdx).getName() + ", score = " + selectedScores[0]);

        //
        // Select features in max CMIM order using fast algorithm
        for (int i = 1; i < max; i++) {
            double curMaxVal = -1.0;
            int curMaxIdx = -1;
            for (int j = 0; j < numFeatures; j++) {
                if (unselectedFeatures[j]) {
                    while ((miCache[j] > curMaxVal) && (idxCache[j] < i)) {
                        double newVal = data.cmi(j, selectedFeatures[idxCache[j]]);
                        if (newVal < miCache[j]) {
                            miCache[j] = newVal;
                        }
                        idxCache[j]++;
                    }
                    if (miCache[j] > curMaxVal) {
                        curMaxVal = miCache[j];
                        curMaxIdx = j;
                    }
                }
            }
            selectedFeatures[i] = curMaxIdx;
            selectedScores[i] = curMaxVal;
            unselectedFeatures[curMaxIdx] = false;
            logger.log(Level.INFO,"Itr " + i + ": selected feature " + fmap.get(curMaxIdx).getName() + ", score = " + curMaxVal);
        }

        ArrayList<String> names = new ArrayList<>();
        ArrayList<Double> scores = new ArrayList<>();
        for (int i = 0; i < max; i++) {
            names.add(fmap.get(selectedFeatures[i]).getName());
            scores.add(selectedScores[i]);
        }

        FeatureSetProvenance provenance = new FeatureSetProvenance(SelectedFeatureSet.class.getName(),dataset.getProvenance(),getProvenance());
        return new SelectedFeatureSet(names,scores,isOrdered(),provenance);
    }

    @Override
    public FeatureSelectorProvenance getProvenance() {
        return new FeatureSelectorProvenanceImpl(this);
    }
}
