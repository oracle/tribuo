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

package org.tribuo.classification.libsvm;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.provenance.ModelProvenance;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A classification model that uses an underlying LibSVM model to make the
 * predictions.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * for the nu-svc algorithm:
 * <pre>
 * Schölkopf B, Smola A, Williamson R, Bartlett P L.
 * "New support vector algorithms"
 * Neural Computation, 2000, 1207-1245.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public class LibSVMClassificationModel extends LibSVMModel<Label> {
    private static final long serialVersionUID = 3L;

    /**
     * This is used when the model hasn't seen as many outputs as the OutputInfo says are there.
     * It stores the unseen labels to ensure the predict method has the right number of outputs.
     * If there are no unobserved labels it's set to Collections.emptySet.
     */
    private final Set<Label> unobservedLabels;

    LibSVMClassificationModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap, List<svm_model> models) {
        super(name, description, featureIDMap, labelIDMap, models.get(0).param.probability == 1, models);
        // This sets up the unobservedLabels variable.
        int[] curLabels = models.get(0).label;
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

    /**
     * Returns the number of support vectors.
     * @return The number of support vectors.
     */
    public int getNumberOfSupportVectors() {
        return models.get(0).SV.length;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        svm_model model = models.get(0);
        svm_node[] features = LibSVMTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set
        if (features.length == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        int[] labels = model.label;
        double[] scores = new double[labels.length];
        if (generatesProbabilities) {
            svm.svm_predict_probability(model, features, scores);
        } else {
            //LibSVM returns a one vs one result, and unpacks it into a score vector by voting
            double[] onevone = new double[labels.length * (labels.length - 1) / 2];
            svm.svm_predict_values(model, features, onevone);
            int counter = 0;
            for (int i = 0; i < labels.length; i++) {
                for (int j = i+1; j < labels.length; j++) {
                    if (onevone[counter] > 0) {
                        scores[i]++;
                    } else {
                        scores[j]++;
                    }
                    counter++;
                }
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
        return new Prediction<>(maxLabel, map, features.length, example, generatesProbabilities);
    }

    @Override
    protected LibSVMClassificationModel copy(String newName, ModelProvenance newProvenance) {
        return new LibSVMClassificationModel(newName,newProvenance,featureIDMap,outputIDInfo,Collections.singletonList(LibSVMModel.copyModel(models.get(0))));
    }

}
