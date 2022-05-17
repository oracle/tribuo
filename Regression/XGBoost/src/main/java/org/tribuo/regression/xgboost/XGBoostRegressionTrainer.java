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

package org.tribuo.regression.xgboost;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Trainer;
import org.tribuo.common.xgboost.XGBoostModel;
import org.tribuo.common.xgboost.XGBoostTrainer;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.provenance.impl.TrainerProvenanceImpl;
import org.tribuo.regression.ImmutableRegressionInfo;
import org.tribuo.regression.Regressor;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link Trainer} which wraps the XGBoost training procedure.
 * This only exposes a few of XGBoost's training parameters.
 * It uses pthreads outside of the JVM to parallelise the computation.
 * <p>
 * Each output dimension is trained independently (and so contains a separate XGBoost ensemble).
 * <p>
 * See:
 * <pre>
 * Chen T, Guestrin C.
 * "XGBoost: A Scalable Tree Boosting System"
 * Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Friedman JH.
 * "Greedy Function Approximation: a Gradient Boosting Machine"
 * Annals of statistics, 2001.
 * </pre>
 * <p>
 * N.B.: XGBoost4J wraps the native C implementation of xgboost that links to various C libraries, including libgomp
 * and glibc (on Linux). If you're running on Alpine, which does not natively use glibc, you'll need to install glibc
 * into the container.
 * On the macOS binary on Maven Central is compiled without
 * OpenMP support, meaning that XGBoost is single threaded on macOS. You can recompile the macOS binary with
 * OpenMP support after installing libomp from homebrew if necessary.
 */
public final class XGBoostRegressionTrainer extends XGBoostTrainer<Regressor> {

    private static final Logger logger = Logger.getLogger(XGBoostRegressionTrainer.class.getName());

    /**
     * Types of regression loss.
     */
    public enum RegressionType {
        /**
         * Squared error loss function.
         */
        LINEAR("reg:squarederror"),
        /**
         * Gamma loss function.
         */
        GAMMA("reg:gamma"),
        /**
         * Tweedie loss function.
         */
        TWEEDIE("reg:tweedie"),
        /**
         * Pseudo-huber loss, a differentiable approximation to absolute error
         */
        PSEUDOHUBER("reg:pseudohubererror");

        /**
         * The parameter name used by the XGBoost native library.
         */
        public final String paramName;

        RegressionType(String paramName) {
            this.paramName = paramName;
        }
    }

    @Config(description="The type of regression.")
    private RegressionType rType = RegressionType.LINEAR;

    /**
     * Creates an XGBoostRegressionTrainer using the default parameters, the squared error loss
     * and the supplied number of trees.
     * @param numTrees The number of trees.
     */
    public XGBoostRegressionTrainer(int numTrees) {
        this(RegressionType.LINEAR, numTrees);
    }

    /**
     * Creates an XGBoostRegressionTrainer using the default parameters, the supplied loss
     * and the supplied number of trees.
     * @param rType The regression loss function.
     * @param numTrees The number of trees.
     */
    public XGBoostRegressionTrainer(RegressionType rType, int numTrees) {
        this(rType, numTrees, 0.3, 0, 6, 1, 1, 1, 1, 0, 4, true, Trainer.DEFAULT_SEED);
    }

    /**
     * Creates an XGBoostRegressionTrainer using the default parameters with the supplied
     * loss, number of trees, number of threads, and logging level.
     * @param rType The regression loss function.
     * @param numTrees The number of trees.
     * @param numThreads The number of threads.
     * @param silent Silence the XGBoost logger.
     */
    public XGBoostRegressionTrainer(RegressionType rType, int numTrees, int numThreads, boolean silent) {
        this(rType, numTrees, 0.3, 0, 6, 1, 1, 1, 1, 0, numThreads, silent, Trainer.DEFAULT_SEED);
    }

    /**
     * Create an XGBoost trainer.
     *
     * @param rType The type of regression to build.
     * @param numTrees Number of trees to boost.
     * @param eta Step size shrinkage parameter (default 0.3, range [0,1]).
     * @param gamma Minimum loss reduction to make a split (default 0, range
     * [0,inf]).
     * @param maxDepth Maximum tree depth (default 6, range [1,inf]).
     * @param minChildWeight Minimum sum of instance weights needed in a leaf
     * (default 1, range [0, inf]).
     * @param subsample Subsample size for each tree (default 1, range (0,1]).
     * @param featureSubsample Subsample features for each tree (default 1,
     * range (0,1]).
     * @param lambda L2 regularization term on weights (default 1).
     * @param alpha L1 regularization term on weights (default 0).
     * @param nThread Number of threads to use (default 4).
     * @param silent Silence the training output text.
     * @param seed RNG seed.
     */
    public XGBoostRegressionTrainer(RegressionType rType, int numTrees, double eta, double gamma, int maxDepth, double minChildWeight, double subsample, double featureSubsample, double lambda, double alpha, int nThread, boolean silent, long seed) {
        super(numTrees,eta,gamma,maxDepth,minChildWeight,subsample,featureSubsample,lambda,alpha,nThread,silent,seed);
        this.rType = rType;

        postConfig();
    }

    /**
     * Create an XGBoost trainer.
     *
     * @param boosterType The base learning algorithm.
     * @param treeMethod The tree building algorithm if using a tree booster.
     * @param rType The type of regression to build.
     * @param numTrees Number of trees to boost.
     * @param eta Step size shrinkage parameter (default 0.3, range [0,1]).
     * @param gamma Minimum loss reduction to make a split (default 0, range
     * [0,inf]).
     * @param maxDepth Maximum tree depth (default 6, range [1,inf]).
     * @param minChildWeight Minimum sum of instance weights needed in a leaf
     * (default 1, range [0, inf]).
     * @param subsample Subsample size for each tree (default 1, range (0,1]).
     * @param featureSubsample Subsample features for each tree (default 1,
     * range (0,1]).
     * @param lambda L2 regularization term on weights (default 1).
     * @param alpha L1 regularization term on weights (default 0).
     * @param nThread Number of threads to use (default 4).
     * @param verbosity Set the logging verbosity of the native library.
     * @param seed RNG seed.
     */
    public XGBoostRegressionTrainer(BoosterType boosterType, TreeMethod treeMethod, RegressionType rType, int numTrees, double eta, double gamma, int maxDepth, double minChildWeight, double subsample, double featureSubsample, double lambda, double alpha, int nThread, LoggingVerbosity verbosity, long seed) {
        super(boosterType,treeMethod,numTrees,eta,gamma,maxDepth,minChildWeight,subsample,featureSubsample,lambda,alpha,nThread,verbosity,seed);
        this.rType = rType;

        postConfig();
    }

    /**
     * This gives direct access to the XGBoost parameter map.
     * <p>
     * It lets you pick things that we haven't exposed like dropout trees, binary classification etc.
     * <p>
     * This sidesteps the validation that Tribuo provides for the hyperparameters, and so can produce unexpected results.
     * @param rType The type of the regression.
     * @param numTrees Number of trees to boost.
     * @param parameters A map from string to object, where object can be Number or String.
     */
    public XGBoostRegressionTrainer(RegressionType rType, int numTrees, Map<String,Object> parameters) {
        super(numTrees,parameters);
        this.rType = rType;
        postConfig();
    }

    /**
     * For olcut.
     */
    private XGBoostRegressionTrainer() { }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        parameters.put("objective",rType.paramName);
        if (!overrideParameters.isEmpty() && !overrideParameters.get("objective").equals(rType.paramName)) {
            throw new PropertyException("","overrideParameters","The objective in overrideParameters must match the one supplied as rType.");
        }
    }

    @Override
    public synchronized XGBoostModel<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance) {
        return train(examples, runProvenance, INCREMENT_INVOCATION_COUNT);
    }

    @Override
    public synchronized XGBoostModel<Regressor> train(Dataset<Regressor> examples, Map<String, Provenance> runProvenance, int invocationCount) {
        if (examples.getOutputInfo().getUnknownCount() > 0) {
            throw new IllegalArgumentException("The supplied Dataset contained unknown Outputs, and this Trainer is supervised.");
        }
        ImmutableFeatureMap featureMap = examples.getFeatureIDMap();
        ImmutableOutputInfo<Regressor> outputInfo = examples.getOutputIDInfo();
        int numOutputs = outputInfo.size();
        if(invocationCount != INCREMENT_INVOCATION_COUNT) {
            setInvocationCount(invocationCount);
        }
        TrainerProvenance trainerProvenance = getProvenance();
        trainInvocationCounter++;
        List<Booster> models = new ArrayList<>();
        try {
            // Use a null response extractor as we'll do the per dimension regression extraction later.
            DMatrixTuple<Regressor> trainingData = convertExamples(examples, featureMap, null);

            // Map the natural order into ids
            int[] dimensionIds = ((ImmutableRegressionInfo) outputInfo).getNaturalOrderToIDMapping();

            // Extract the weights and the regression targets.
            float[][] outputs = new float[numOutputs][examples.size()];
            float[] weights = new float[examples.size()];
            int i = 0;
            for (Example<Regressor> e : examples) {
                weights[i] = e.getWeight();
                double[] curOutputs = e.getOutput().getValues();
                // Transpose them for easy training.
                for (int j = 0; j < numOutputs; j++) {
                    outputs[dimensionIds[j]][i] = (float) curOutputs[j];
                }
                i++;
            }
            trainingData.data.setWeight(weights);

            Map<String,Object> curParams = overrideParameters.isEmpty() ? copyParams(parameters) : copyParams(overrideParameters);
            // Finished setup, now train one model per dimension.
            for (i = 0; i < numOutputs; i++) {
                trainingData.data.setLabel(outputs[i]);
                models.add(XGBoost.train(trainingData.data, curParams, numTrees, Collections.emptyMap(), null, null));
            }
        } catch (XGBoostError e) {
            logger.log(Level.SEVERE, "XGBoost threw an error", e);
            throw new IllegalStateException(e);
        }

        ModelProvenance provenance = new ModelProvenance(XGBoostModel.class.getName(), OffsetDateTime.now(), examples.getProvenance(), trainerProvenance, runProvenance);
        XGBoostModel<Regressor> xgModel = createModel("xgboost-regression-model", provenance, featureMap, outputInfo, models, new XGBoostRegressionConverter());

        return xgModel;
    }

    @Override
    public TrainerProvenance getProvenance() {
        return new TrainerProvenanceImpl(this);
    }
}
