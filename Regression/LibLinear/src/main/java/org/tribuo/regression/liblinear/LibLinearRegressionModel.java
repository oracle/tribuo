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

package org.tribuo.regression.liblinear;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps a LibLinear-java model.
 * <p>
 * It disables the LibLinear debug output as it's very chatty.
 * <p>
 * It contains an independent liblinear model for each regression dimension.
 * <p>
 * See:
 * <pre>
 * Fan RE, Chang KW, Hsieh CJ, Wang XR, Lin CJ.
 * "LIBLINEAR: A library for Large Linear Classification"
 * Journal of Machine Learning Research, 2008.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public class LibLinearRegressionModel extends LibLinearModel<Regressor> {
    private static final long serialVersionUID = 2L;

    private static final Logger logger = Logger.getLogger(LibLinearRegressionModel.class.getName());

    private final String[] dimensionNames;

    // Not final as it doesn't exist in 4.0 or 4.1 and so must be created on deserialization.
    private int[] mapping;

    LibLinearRegressionModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputInfo, List<de.bwaldvogel.liblinear.Model> models) {
        super(name, description, featureIDMap, outputInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputInfo);
        this.mapping = ((ImmutableRegressionInfo) outputInfo).getIDtoNaturalOrderMapping();
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        FeatureNode[] features = LibLinearTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set
        if (features.length == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        double[] scores = new double[models.get(0).getNrClass()];
        double[] regressedValues = new double[models.size()];

        // Map through the id -> regressor dimension natural order (i.e., lexicographic) to ensure the regressor is
        // constructed correctly.
        for (int i = 0; i < regressedValues.length; i++) {
            regressedValues[mapping[i]] = Linear.predictValues(models.get(i), features, scores);
        }

        Regressor regressor = new Regressor(dimensionNames,regressedValues);
        return new Prediction<>(regressor, features.length - 1, example);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;
        double[][] featureWeights = getFeatureWeights();

        Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);

        for (int i = 0; i < featureWeights.length; i++) {
            // Exclude bias
            int numFeatures = featureWeights[i].length - 1;
            for (int j = 0; j < numFeatures; j++) {
                Pair<String, Double> cur = new Pair<>(featureIDMap.get(j).getName(), featureWeights[i][j]);
                if (maxFeatures < 0 || q.size() < maxFeatures) {
                    q.offer(cur);
                } else if (comparator.compare(cur, q.peek()) > 0) {
                    q.poll();
                    q.offer(cur);
                }
            }
            List<Pair<String, Double>> list = new ArrayList<>();
            while (q.size() > 0) {
                list.add(q.poll());
            }
            Collections.reverse(list);
            map.put(dimensionNames[mapping[i]], list);
        }

        return map;
    }

    @Override
    protected LibLinearRegressionModel copy(String newName, ModelProvenance newProvenance) {
        List<de.bwaldvogel.liblinear.Model> newModels = new ArrayList<>();
        for (de.bwaldvogel.liblinear.Model m : models) {
            newModels.add(copyModel(m));
        }
        return new LibLinearRegressionModel(newName,newProvenance,featureIDMap,outputIDInfo,newModels);
    }

    @Override
    protected double[][] getFeatureWeights() {
        double[][] featureWeights = new double[models.size()][];

        for (int i = 0; i < models.size(); i++) {
            featureWeights[i] = models.get(i).getFeatureWeights();
        }

        return featureWeights;
    }

    /**
     * The call to model.getFeatureWeights in the public methods copies the
     * weights array so this inner method exists to save the copy in getExcuses.
     * <p>
     * If it becomes a problem then we could cache the feature weights in the
     * model.
     *
     * @param e The example.
     * @param allFeatureWeights The feature weights.
     * @return An excuse for this example.
     */
    @Override
    protected Excuse<Regressor> innerGetExcuse(Example<Regressor> e, double[][] allFeatureWeights) {
        Prediction<Regressor> prediction = predict(e);
        Map<String, List<Pair<String, Double>>> weightMap = new HashMap<>();
        for (int i = 0; i < allFeatureWeights.length; i++) {
            List<Pair<String, Double>> scores = new ArrayList<>();
            for (Feature f : e) {
                int id = featureIDMap.getID(f.getName());
                if (id > -1) {
                    double score = allFeatureWeights[i][id] * f.getValue();
                    scores.add(new Pair<>(f.getName(), score));
                }
            }
            scores.sort((o1, o2) -> o2.getB().compareTo(o1.getB()));
            weightMap.put(dimensionNames[mapping[i]], scores);
        }

        return new Excuse<>(e, prediction, weightMap);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        // Add mapping field to 4.0, 4.1 models and rearrange the dimensions.
        if (mapping == null) {
            this.mapping = ((ImmutableRegressionInfo) outputIDInfo).getIDtoNaturalOrderMapping();
            List<de.bwaldvogel.liblinear.Model> newModels = new ArrayList<>(this.models);

            for (int i = 0; i < mapping.length; i++) {
                newModels.set(i,this.models.get(mapping[i]));
            }

            this.models = Collections.unmodifiableList(newModels);
        }
    }
}
