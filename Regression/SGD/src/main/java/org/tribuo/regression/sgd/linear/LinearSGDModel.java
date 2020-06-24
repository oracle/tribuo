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

package org.tribuo.regression.sgd.linear;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;

/**
 * The inference time version of a linear model trained using SGD.
 * The output dimensions are independent, unless they are tied together by the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends Model<Regressor> {
    private static final long serialVersionUID = 3L;

    private final String[] dimensionNames;
    private final DenseMatrix weights;

    LinearSGDModel(String name, String[] dimensionNames, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap,
                          LinearParameters parameters) {
        super(name, description, featureIDMap, labelIDMap, false);
        this.weights = parameters.getWeightMatrix();
        this.dimensionNames = dimensionNames;
    }

    private LinearSGDModel(String name, String[] dimensionNames, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> labelIDMap,
                          DenseMatrix weights) {
        super(name, description, featureIDMap, labelIDMap, false);
        this.weights = weights;
        this.dimensionNames = dimensionNames;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        SparseVector features = SparseVector.createSparseVector(example,featureIDMap,true);
        if (features.numActiveElements() == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        DenseVector prediction = weights.leftMultiply(features);
        return new Prediction<>(new Regressor(dimensionNames,prediction.toArray()), features.numActiveElements(), example);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() + 1 : n;

        Comparator<Pair<String,Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));

        //
        // Use a priority queue to find the top N features.
        int numClasses = weights.getDimension1Size();
        int numFeatures = weights.getDimension2Size()-1; //Removing the bias feature.
        Map<String, List<Pair<String,Double>>> map = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures, comparator);

            for (int j = 0; j < numFeatures; j++) {
                Pair<String,Double> curr = new Pair<>(featureIDMap.get(j).getName(), weights.get(i,j));

                if (q.size() < maxFeatures) {
                    q.offer(curr);
                } else if (comparator.compare(curr, q.peek()) > 0) {
                    q.poll();
                    q.offer(curr);
                }
            }
            Pair<String,Double> curr = new Pair<>(BIAS_FEATURE, weights.get(i,numFeatures));

            if (q.size() < maxFeatures) {
                q.offer(curr);
            } else if (comparator.compare(curr, q.peek()) > 0) {
                q.poll();
                q.offer(curr);
            }
            ArrayList<Pair<String,Double>> b = new ArrayList<>();
            while (q.size() > 0) {
                b.add(q.poll());
            }

            Collections.reverse(b);
            map.put(dimensionNames[i], b);
        }
        return map;
    }

    @Override
    public Optional<Excuse<Regressor>> getExcuse(Example<Regressor> example) {
        Prediction<Regressor> prediction = predict(example);
        Map<String, List<Pair<String, Double>>> weightMap = new HashMap<>();
        int numOutputs = weights.getDimension1Size();
        int numFeatures = weights.getDimension2Size()-1; //Remove bias feature

        for (int i = 0; i < numOutputs; i++) {
            List<Pair<String, Double>> classScores = new ArrayList<>();
            for (Feature f : example) {
                int id = featureIDMap.getID(f.getName());
                if (id > -1) {
                    double score = weights.get(i,id) * f.getValue();
                    classScores.add(new Pair<>(f.getName(), score));
                }
            }
            classScores.add(new Pair<>(BIAS_FEATURE, weights.get(i,numFeatures)));
            classScores.sort((Pair<String, Double> o1, Pair<String, Double> o2) -> o2.getB().compareTo(o1.getB()));
            weightMap.put(dimensionNames[i], classScores);
        }

        return Optional.of(new Excuse<>(example, prediction, weightMap));
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,Arrays.copyOf(dimensionNames,dimensionNames.length),newProvenance,featureIDMap,outputIDInfo,getWeightsCopy());
    }

    public DenseMatrix getWeightsCopy() {
        return weights.copy();
    }
}
