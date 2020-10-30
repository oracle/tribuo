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

import com.oracle.labs.mlrg.olcut.config.Config;
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
import org.tribuo.util.Util;

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

    @Config(description="Standardise the regression outputs before training.")
    private boolean standardize = false;

    /**
     * For olcut.
     */
    protected LibSVMRegressionTrainer() {}

    /**
     * Constructs a LibSVMRegressionTrainer using the supplied parameters without standardizing the regression variables.
     * @param parameters The SVM parameters.
     */
    public LibSVMRegressionTrainer(SVMParameters<Regressor> parameters) {
        this(parameters, false);
    }

    /**
     * Constructs a LibSVMRegressionTrainer using the supplied parameters.
     * @param parameters The SVM parameters.
     * @param standardize Standardize the regression outputs before training.
     */
    public LibSVMRegressionTrainer(SVMParameters<Regressor> parameters, boolean standardize) {
        super(parameters);
        this.standardize = standardize;
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
        if (models.get(0) instanceof ModelWithMeanVar) {
            // models have been standardized, unpick and use standardized constructor
            double[] means = new double[models.size()];
            double[] variances = new double[models.size()];
            List<svm_model> unpickedModels = new ArrayList<>(models.size());
            for (int i = 0; i < models.size(); i++) {
                ModelWithMeanVar curModel = (ModelWithMeanVar) models.get(0);
                means[i] = curModel.mean;
                variances[i] = curModel.variance;
                unpickedModels.add(curModel.innerModel);
            }
            return new LibSVMRegressionModel("svm-regression-model", provenance, featureIDMap, outputIDInfo, unpickedModels, means, variances);
        } else {
            return new LibSVMRegressionModel("svm-regression-model", provenance, featureIDMap, outputIDInfo, models);
        }
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
            if (standardize) {
                Pair<Double,Double> meanVar = Util.meanAndVariance(outputs[i]);
                double mean = meanVar.getA();
                double variance = meanVar.getB();
                Util.standardizeInPlace(outputs[i],mean,variance);
                String checkString = svm.svm_check_parameter(problem, curParams);
                if(checkString != null) {
                    throw new IllegalArgumentException("Error checking SVM parameters: " + checkString);
                }
                svm_model trained = svm.svm_train(problem, curParams);
                models.add(new ModelWithMeanVar(trained, mean, variance));
            } else {
                String checkString = svm.svm_check_parameter(problem, curParams);
                if(checkString != null) {
                    throw new IllegalArgumentException("Error checking SVM parameters: " + checkString);
                }
                models.add(svm.svm_train(problem, curParams));
            }
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

    /**
     * Wrapper class to pass the means & variances through with the svm_models
     * <p>
     * Should not be persisted outside of the trainer.
     */
    private static class ModelWithMeanVar extends svm_model {
        final double mean;
        final double variance;
        final svm_model innerModel;

        /**
         * Note this doesn't copy the model, it essentially wraps it.
         * @param model The model to wrap.
         * @param mean The mean.
         * @param variance The variance.
         */
        private ModelWithMeanVar(svm_model model, double mean, double variance) {
            this.mean = mean;
            this.variance = variance;

            this.param = model.param;
            this.l = model.l;
            this.nr_class = model.nr_class;
            this.rho = model.rho;
            this.probA = model.probA;
            this.probB = model.probB;
            this.label = model.label;
            this.sv_indices = model.sv_indices;
            this.nSV = model.nSV;
            this.SV = model.SV;
            this.sv_coef = model.sv_coef;

            this.innerModel = model;
        }
    }
}
