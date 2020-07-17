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

package org.tribuo.classification.liblinear;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.provenance.ModelProvenance;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps a LibLinear-java classification model.
 * <p>
 * It disables the LibLinear debug output as it's very chatty.
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
public class LibLinearClassificationModel extends LibLinearModel<Label> {
    private static final long serialVersionUID = 3L;

    private static final Logger logger = Logger.getLogger(LibLinearClassificationModel.class.getName());

    /**
     * This is used when the model hasn't seen as many outputs as the OutputInfo says are there.
     * It stores the unseen labels to ensure the predict method has the right number of outputs.
     * If there are no unobserved labels it's set to Collections.emptySet.
     */
    private final Set<Label> unobservedLabels;

    LibLinearClassificationModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap, List<de.bwaldvogel.liblinear.Model> models) {
        super(name, description, featureIDMap, labelIDMap, models.get(0).isProbabilityModel(), models);
        // This sets up the unobservedLabels variable.
        int[] curLabels = models.get(0).getLabels();
        if (curLabels.length != labelIDMap.size()) {
            Map<Integer,Label> tmp = new HashMap<>();
            for (Pair<Integer,Label> p : labelIDMap) {
                tmp.put(p.getA(),p.getB());
            }
            for (int i = 0; i < curLabels.length; i++) {
                tmp.remove(i);
            }
            Set<Label> tmpSet = new HashSet<>(tmp.values().size());
            for (Label l : tmp.values()) {
                tmpSet.add(new Label(l.getLabel(),0.0));
            }
            this.unobservedLabels = Collections.unmodifiableSet(tmpSet);
        } else {
            this.unobservedLabels = Collections.emptySet();
        }
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        FeatureNode[] features = LibLinearTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set
        if (features.length == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        de.bwaldvogel.liblinear.Model model = models.get(0);

        int[] labels = model.getLabels();
        double[] scores = new double[labels.length];

        if (model.isProbabilityModel()) {
            Linear.predictProbability(model, features, scores);
        } else {
            Linear.predictValues(model, features, scores);
            if ((model.getNrClass() == 2) && (scores[1] == 0.0)) {
                scores[1] = -scores[0];
            }
        }

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> map = new LinkedHashMap<>();
        for (int i = 0; i < scores.length; i++) {
            String name = outputIDInfo.getOutput(labels[i]).getLabel();
            Label label = new Label(name, scores[i]);
            map.put(name,label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }
        if (!unobservedLabels.isEmpty()) {
            for (Label l : unobservedLabels) {
                map.put(l.getLabel(),l);
            }
        }
        return new Prediction<>(maxLabel, map, features.length-1, example, generatesProbabilities);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;
        de.bwaldvogel.liblinear.Model model = models.get(0);
        int[] labels = model.getLabels();
        double[] featureWeights = model.getFeatureWeights();

        Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
        
        /*
         * Liblinear stores its weights as follows
         * +------------------+------------------+------------+
         * | nr_class weights | nr_class weights |  ...
         * | for 1st feature  | for 2nd feature  |
         * +------------------+------------------+------------+
         *
         * If bias &gt;= 0, x becomes [x; bias]. The number of features is
         * increased by one, so w is a (nr_feature+1)*nr_class array. The
         * value of bias is stored in the variable bias.
         */

        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        int numClasses = model.getNrClass();
        int numFeatures = model.getNrFeature();
        if (numClasses == 2) {
            //
            // When numClasses == 2, liblinear only stores one set of weights.
            PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);

            for (int i = 0; i < numFeatures; i++) {
                Pair<String, Double> cur = new Pair<>(featureIDMap.get(i).getName(), featureWeights[i]);
                if (q.size() < maxFeatures) {
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
            map.put(outputIDInfo.getOutput(labels[0]).getLabel(), list);

            List<Pair<String, Double>> otherList = new ArrayList<>();
            for (Pair<String, Double> f : list) {
                Pair<String, Double> otherF = new Pair<>(f.getA(), -f.getB());
                otherList.add(otherF);
            }
            map.put(outputIDInfo.getOutput(labels[1]).getLabel(), otherList);
        } else {
            for (int i = 0; i < labels.length; i++) {
                PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);
                //iterate over the non-bias features
                for (int j = 0; j < numFeatures; j++) {
                    int index = (j * numClasses) + i;
                    Pair<String, Double> cur = new Pair<>(featureIDMap.get(j).getName(), featureWeights[index]);
                    if (q.size() < maxFeatures) {
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
                map.put(outputIDInfo.getOutput(labels[i]).getLabel(), list);
            }
        }
        return map;
    }

    @Override
    protected LibLinearClassificationModel copy(String newName, ModelProvenance newProvenance) {
        return new LibLinearClassificationModel(newName,newProvenance,featureIDMap,outputIDInfo,Collections.singletonList(copyModel(models.get(0))));
    }

    @Override
    protected double[][] getFeatureWeights() {
        double[][] featureWeights = new double[1][];
        featureWeights[0] = models.get(0).getFeatureWeights();
        return featureWeights;
    }

    /**
     * The call to model.getFeatureWeights in the public methods copies the
     * weights array so this inner method exists to save the copy in getExcuses.
     * <p>
     * If it becomes a problem then we could cache the feature weights in the
     * model.
     * @param e The example.
     * @param allFeatureWeights The feature weights.
     * @return An excuse for this example.
     */
    @Override
    protected Excuse<Label> innerGetExcuse(Example<Label> e, double[][] allFeatureWeights) {
        de.bwaldvogel.liblinear.Model model = models.get(0);
        double[] featureWeights = allFeatureWeights[0];
        int[] labels = model.getLabels();
        int numClasses = model.getNrClass();

        Prediction<Label> prediction = predict(e);
        Map<String, List<Pair<String, Double>>> weightMap = new HashMap<>();

        if (numClasses == 2) {
            List<Pair<String, Double>> posScores = new ArrayList<>();
            List<Pair<String, Double>> negScores = new ArrayList<>();
            for (Feature f : e) {
                int id = featureIDMap.getID(f.getName());
                if (id > -1) {
                    double score = featureWeights[id] * f.getValue();
                    posScores.add(new Pair<>(f.getName(), score));
                    negScores.add(new Pair<>(f.getName(), -score));
                }
            }
            posScores.sort((o1, o2) -> o2.getB().compareTo(o1.getB()));
            negScores.sort((o1, o2) -> o2.getB().compareTo(o1.getB()));
            weightMap.put(outputIDInfo.getOutput(labels[0]).getLabel(),posScores);
            weightMap.put(outputIDInfo.getOutput(labels[1]).getLabel(),negScores);
        } else {
            for (int i = 0; i < labels.length; i++) {
                List<Pair<String, Double>> classScores = new ArrayList<>();
                for (Feature f : e) {
                    int id = featureIDMap.getID(f.getName());
                    if (id > -1) {
                        double score = featureWeights[id * numClasses + i] * f.getValue();
                        classScores.add(new Pair<>(f.getName(), score));
                    }
                }
                classScores.sort((Pair<String, Double> o1, Pair<String, Double> o2) -> o2.getB().compareTo(o1.getB()));
                weightMap.put(outputIDInfo.getOutput(labels[i]).getLabel(), classScores);
            }
        }

        return new Excuse<>(e, prediction, weightMap);
    }
}
