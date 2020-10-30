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

package org.tribuo.regression.libsvm;

import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Prediction;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A regression model that uses an underlying libSVM model to make the
 * predictions. Contains an independent model for each output dimension.
 * <p>
 * See:
 * <pre>
 * Chang CC, Lin CJ.
 * "LIBSVM: a library for Support Vector Machines"
 * ACM transactions on intelligent systems and technology (TIST), 2011.
 * </pre>
 * for the nu-svr algorithm:
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
public class LibSVMRegressionModel extends LibSVMModel<Regressor> {
    private static final long serialVersionUID = 2L;

    private final String[] dimensionNames;

    private final double[] means;

    private final double[] variances;

    private final boolean standardized;

    /**
     * Constructs a LibSVMRegressionModel with regular outputs.
     * @param name The model name.
     * @param description The model provenance.
     * @param featureIDMap The features this model knows about.
     * @param outputIDInfo The outputs this model can produce.
     * @param models The svm_models themselves.
     */
    LibSVMRegressionModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<svm_model> models) {
        super(name, description, featureIDMap, outputIDInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputIDInfo);
        this.means = null;
        this.variances = null;
        this.standardized = false;
    }

    /**
     * Constructs a LibSVMRegressionModel with standardized outputs that must be upscaled during prediction.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The features this model knows about.
     * @param outputIDInfo The outputs this model can produce.
     * @param models The svm_models themselves.
     * @param means The output dimension means.
     * @param variances The output dimension variances.
     */
    LibSVMRegressionModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<svm_model> models, double[] means, double[] variances) {
        super(name, provenance, featureIDMap, outputIDInfo, false, models);
        this.dimensionNames = Regressor.extractNames(outputIDInfo);
        this.means = means;
        this.variances = variances;
        this.standardized = true;
    }

    /**
     * Returns the support vectors used for each dimension.
     * @return The support vectors.
     */
    public Map<String,Integer> getNumberOfSupportVectors() {
        Map<String,Integer> output = new HashMap<>();

        for (int i = 0; i < dimensionNames.length; i++) {
            output.put(dimensionNames[i],models.get(i).SV.length);
        }

        return output;
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        svm_node[] features = LibSVMTrainer.exampleToNodes(example, featureIDMap, null);
        if (features.length == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        double[] scores = new double[1];
        double[] regressedValues = new double[models.size()];

        for (int i = 0; i < regressedValues.length; i++) {
            regressedValues[i] = svm.svm_predict_values(models.get(i), features, scores);
            if (standardized) {
                regressedValues[i] = (regressedValues[i] * variances[i]) + means[i];
            }
        }

        Regressor regressor = new Regressor(dimensionNames,regressedValues);
        return new Prediction<>(regressor, features.length, example);
    }

    @Override
    protected LibSVMRegressionModel copy(String newName, ModelProvenance newProvenance) {
        List<svm_model> newModels = new ArrayList<>();
        for (svm_model m : models) {
            newModels.add(copyModel(m));
        }
        return new LibSVMRegressionModel(newName,newProvenance,featureIDMap,outputIDInfo,newModels);
    }

}
