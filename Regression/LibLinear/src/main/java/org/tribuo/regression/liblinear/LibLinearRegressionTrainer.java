/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.liblinear.LinearRegressionType.LinearType;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which wraps a liblinear-java regression trainer.
 * <p>
 * This generates an independent liblinear model for each regression dimension.
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
public class LibLinearRegressionTrainer extends LibLinearTrainer<Regressor> {

    private static final Logger logger = Logger.getLogger(LibLinearRegressionTrainer.class.getName());

    /**
     * Used in the tests for regression dimension re-ordering to revert to the 4.2 behaviour.
     */
    boolean forceZero = false;

    /**
     * Creates a trainer using the default values (L2R_L2LOSS_SVR, 1, 1000, 0.1, 0.1).
     */
    public LibLinearRegressionTrainer() {
        this(new LinearRegressionType(LinearType.L2R_L2LOSS_SVR));
    }

    /**
     * Creates a trainer for a LibLinear regression model.
     * <p>
     * Uses the default values of cost = 1.0, maxIterations = 1000, terminationCriterion = 0.1, epsilon = 0.1.
     * @param trainerType Loss function and optimisation method.
     */
    public LibLinearRegressionTrainer(LinearRegressionType trainerType) {
        this(trainerType,1.0,1000,0.1,0.1);
    }

    /**
     * Creates a trainer for a LibLinear regression model.
     * <p>
     * Uses {@link Trainer#DEFAULT_SEED} as the RNG seed.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param epsilon The insensitivity of the regression loss to small differences.
     */
    public LibLinearRegressionTrainer(LinearRegressionType trainerType, double cost, int maxIterations, double terminationCriterion, double epsilon) {
        this(trainerType,cost,maxIterations,terminationCriterion,epsilon,Trainer.DEFAULT_SEED);
    }

    /**
     * Creates a trainer for a LibLinear regression model.
     * @param trainerType Loss function and optimisation method combination.
     * @param cost Cost penalty for each incorrectly classified training point.
     * @param maxIterations The maximum number of dataset iterations.
     * @param terminationCriterion How close does the optimisation function need to be before terminating that subproblem (usually set to 0.1).
     * @param epsilon The insensitivity of the regression loss to small differences.
     * @param seed The RNG seed.
     */
    public LibLinearRegressionTrainer(LinearRegressionType trainerType, double cost, int maxIterations, double terminationCriterion, double epsilon, long seed) {
        super(trainerType,cost,maxIterations,terminationCriterion,epsilon,seed);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (!trainerType.isRegression()) {
            throw new IllegalArgumentException("Supplied classification or anomaly detection parameters to a regression linear model.");
        }
    }

    @Override
    protected List<Model> trainModels(Parameter curParams, int numFeatures, FeatureNode[][] features, double[][] outputs) {
        ArrayList<Model> models = new ArrayList<>();

        for (int i = 0; i < outputs.length; i++) {
            Problem data = new Problem();

            data.l = features.length;
            data.y = outputs[i];
            data.x = features;
            data.n = numFeatures;
            data.bias = 1.0;

            /*
             * Enforces the behaviour of Tribuo 4.2 and liblinear-java 2.43 to allow
             * TestLibLinear.testThreeDenseData to validate that regression indices
             * are handled correctly.
             */
            if (forceZero) {
                curParams.setRandom(new Random(0));
            }
            models.add(Linear.train(data, curParams));
        }

        return models;
    }

    @Override
    protected LibLinearModel<Regressor> createModel(ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo, List<Model> models) {
        if (models.size() != outputIDInfo.size()) {
            throw new IllegalArgumentException("Regression uses one model per dimension. Found " + models.size() + " models, and " + outputIDInfo.size() + " dimensions.");
        }
        return new LibLinearRegressionModel("liblinear-regression-model",provenance,featureIDMap,outputIDInfo,models);
    }

    @Override
    protected Pair<FeatureNode[][], double[][]> extractData(Dataset<Regressor> data, ImmutableOutputInfo<Regressor> outputInfo, ImmutableFeatureMap featureMap) {
        int numOutputs = outputInfo.size();
        int[] dimensionIds = ((ImmutableRegressionInfo) outputInfo).getNaturalOrderToIDMapping();
        List<FeatureNode> featureCache = new ArrayList<>();
        FeatureNode[][] features = new FeatureNode[data.size()][];
        double[][] outputs = new double[numOutputs][data.size()];
        int i = 0;
        for (Example<Regressor> e : data) {
            double[] curOutputs = e.getOutput().getValues();
            for (int j = 0; j < curOutputs.length; j++) {
                outputs[dimensionIds[j]][i] = curOutputs[j];
            }
            features[i] = exampleToNodes(e,featureMap,featureCache);
            i++;
        }
        return new Pair<>(features,outputs);
    }

}
