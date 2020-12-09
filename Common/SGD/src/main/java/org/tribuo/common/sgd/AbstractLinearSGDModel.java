/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.sgd;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;

public abstract class AbstractLinearSGDModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 1L;

    /**
     * The weights for this linear model.
     */
    // Note this is not final to allow backwards compatibility for 4.0 models which need to rewrite the field on load.
    protected DenseMatrix baseWeights;

    /**
     * Constructs a linear model trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param weights The model weights.
     * @param generatesProbabilities Does this model generate probabilities?
     */
    protected AbstractLinearSGDModel(String name, ModelProvenance provenance,
                           ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                           DenseMatrix weights, boolean generatesProbabilities) {
        super(name, provenance, featureIDMap, outputIDInfo, generatesProbabilities);
        this.baseWeights = weights;
    }

    /**
     * Generates the dense vector prediction from the supplied example.
     * @param example The example to use for prediction.
     * @return The prediction and the number of features involved.
     */
    protected PredAndActive predictSingle(Example<T> example) {
        SparseVector features = SparseVector.createSparseVector(example,featureIDMap,true);
        if (features.numActiveElements() == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        return new PredAndActive(baseWeights.leftMultiply(features),features.numActiveElements());
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() + 1 : n;

        Comparator<Pair<String,Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));

        //
        // Use a priority queue to find the top N features.
        int numClasses = baseWeights.getDimension1Size();
        int numFeatures = baseWeights.getDimension2Size()-1; //Removing the bias feature.
        Map<String, List<Pair<String,Double>>> map = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures, comparator);

            for (int j = 0; j < numFeatures; j++) {
                Pair<String,Double> curr = new Pair<>(featureIDMap.get(j).getName(), baseWeights.get(i,j));

                if (q.size() < maxFeatures) {
                    q.offer(curr);
                } else if (comparator.compare(curr, q.peek()) > 0) {
                    q.poll();
                    q.offer(curr);
                }
            }
            Pair<String,Double> curr = new Pair<>(BIAS_FEATURE, baseWeights.get(i,numFeatures));

            if (q.size() < maxFeatures) {
                q.offer(curr);
            } else if (comparator.compare(curr, q.peek()) > 0) {
                q.poll();
                q.offer(curr);
            }
            List<Pair<String,Double>> b = new ArrayList<>();
            while (q.size() > 0) {
                b.add(q.poll());
            }

            Collections.reverse(b);
            map.put(getDimensionName(i), b);
        }
        return map;
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        Prediction<T> prediction = predict(example);
        Map<String, List<Pair<String, Double>>> weightMap = new HashMap<>();
        int numClasses = baseWeights.getDimension1Size();
        int numFeatures = baseWeights.getDimension2Size()-1;

        for (int i = 0; i < numClasses; i++) {
            List<Pair<String, Double>> classScores = new ArrayList<>();
            for (Feature f : example) {
                int id = featureIDMap.getID(f.getName());
                if (id > -1) {
                    double score = baseWeights.get(i,id) * f.getValue();
                    classScores.add(new Pair<>(f.getName(), score));
                }
            }
            classScores.add(new Pair<>(Model.BIAS_FEATURE, baseWeights.get(i,numFeatures)));
            classScores.sort((Pair<String, Double> o1, Pair<String, Double> o2) -> o2.getB().compareTo(o1.getB()));
            weightMap.put(getDimensionName(i), classScores);
        }

        return Optional.of(new Excuse<>(example, prediction, weightMap));
    }

    /**
     * Gets the name of the indexed output dimension.
     * @param index The output dimension index.
     * @return The name of the requested output dimension.
     */
    protected abstract String getDimensionName(int index);

    /**
     * Returns a copy of the weights.
     * @return A copy of the weights.
     */
    public DenseMatrix getWeightsCopy() {
        return baseWeights.copy();
    }

    /**
     * A nominal tuple used to capture the prediction and the number of active features used by the model.
     */
    protected static final class PredAndActive {
        public final DenseVector prediction;
        public final int numActiveFeatures;

        PredAndActive(DenseVector prediction, int numActiveFeatures) {
            this.prediction = prediction;
            this.numActiveFeatures = numActiveFeatures;
        }
    }
}
