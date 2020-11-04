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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * A trainer for regression models that uses LibSVM. Trains an independent model for each output dimension.
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
public class LibSVMRegressionTrainer extends LibSVMTrainer<Regressor> {
    private static final Logger logger = Logger.getLogger(LibSVMRegressionTrainer.class.getName());

    /**
     * For olcut.
     */
    protected LibSVMRegressionTrainer() {}

    public LibSVMRegressionTrainer(SVMParameters<Regressor> parameters) {
        super(parameters);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (!svmType.isRegression()) {
            throw new IllegalArgumentException("Supplied classification or anomaly detection parameters to a regression SVM.");
        }
    }

    @Override
    protected LibSVMModel<Regressor> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<svm_model> models) {
        return new LibSVMRegressionModel("svm-regression-model", provenance, featureIDMap, outputIDInfo, models);
    }

    @Override
    protected List<svm_model> trainModels(svm_parameter curParams, int numFeatures, svm_node[][] features, double[][] outputs) {
        ArrayList<svm_model> models = new ArrayList<>();

        for (int i = 0; i < outputs.length; i++) {
            svm_problem problem = new svm_problem();
            problem.l = outputs[i].length;
            problem.x = features;
            problem.y = outputs[i];
            if (curParams.gamma == 0) {
                curParams.gamma = 1.0 / numFeatures;
            }
            String checkString = svm.svm_check_parameter(problem, curParams);
            if(checkString != null) {
                throw new IllegalArgumentException("Error checking SVM parameters: " + checkString);
            }
            models.add(svm.svm_train(problem, curParams));
        }

        return Collections.unmodifiableList(models);
    }

    @Override
    protected Pair<svm_node[][], double[][]> extractData(Dataset<Regressor> data, ImmutableOutputInfo<Regressor> outputInfo, ImmutableFeatureMap featureMap) {
        int numOutputs = outputInfo.size();
        ArrayList<svm_node> buffer = new ArrayList<>();
        svm_node[][] features = new svm_node[data.size()][];
        double[][] outputs = new double[numOutputs][data.size()];
        int i = 0;
        for (Example<Regressor> e : data) {
            double[] curOutputs = e.getOutput().getValues();
            for (int j = 0; j < curOutputs.length; j++) {
                outputs[j][i] = curOutputs[j];
            }
            features[i] = exampleToNodes(e,featureMap,buffer);
            i++;
        }
        return new Pair<>(features,outputs);
    }
}
