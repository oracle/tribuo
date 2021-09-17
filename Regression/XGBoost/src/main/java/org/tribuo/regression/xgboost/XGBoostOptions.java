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

package org.tribuo.regression.xgboost;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer.RegressionType;

/**
 * CLI options for configuring an XGBoost regression trainer.
 */
public class XGBoostOptions implements Options {
    /**
     * Regression type to use. Defaults to LINEAR.
     */
    @Option(longName = "xgb-regression-metric", usage = "Regression type to use. Defaults to LINEAR.")
    public RegressionType rType = RegressionType.LINEAR;
    /**
     * Number of trees in the ensemble.
     */
    @Option(longName = "xgb-ensemble-size", usage = "Number of trees in the ensemble.")
    public int ensembleSize = -1;
    /**
     * L1 regularization term for weights (default 0).
     */
    @Option(longName = "xgb-alpha", usage = "L1 regularization term for weights (default 0).")
    public float alpha = 0.0f;
    /**
     * Minimum sum of instance weights needed in a leaf (default 1, range [0,inf]).
     */
    @Option(longName = "xgb-min-weight", usage = "Minimum sum of instance weights needed in a leaf (default 1, range [0,inf]).")
    public float minWeight = 1;
    /**
     * Max tree depth (default 6, range (0,inf]).
     */
    @Option(longName = "xgb-max-depth", usage = "Max tree depth (default 6, range (0,inf]).")
    public int depth = 6;
    /**
     * Step size shrinkage parameter (default 0.3, range [0,1]).
     */
    @Option(longName = "xgb-eta", usage = "Step size shrinkage parameter (default 0.3, range [0,1]).")
    public float eta = 0.3f;
    /**
     * Subsample features for each tree (default 1, range (0,1]).
     */
    @Option(longName = "xgb-subsample-features", usage = "Subsample features for each tree (default 1, range (0,1]).")
    public float subsampleFeatures;
    /**
     * Minimum loss reduction to make a split (default 0, range [0,inf]).
     */
    @Option(longName = "xgb-gamma", usage = "Minimum loss reduction to make a split (default 0, range [0,inf]).")
    public float gamma = 0.0f;
    /**
     * L2 regularization term for weights (default 1).
     */
    @Option(longName = "xgb-lambda", usage = "L2 regularization term for weights (default 1).")
    public float lambda = 1.0f;
    /**
     * Make the XGBoost training procedure quiet.
     */
    @Option(longName = "xgb-quiet", usage = "Make the XGBoost training procedure quiet.")
    public boolean quiet;
    /**
     * Subsample size for each tree (default 1, range (0,1]).
     */
    @Option(longName = "xgb-subsample", usage = "Subsample size for each tree (default 1, range (0,1]).")
    public float subsample = 1.0f;
    /**
     * Number of threads to use (default 4, range (1, num hw threads)).
     */
    @Option(longName = "xgb-num-threads", usage = "Number of threads to use (default 4, range (1, num hw threads)).")
    public int numThreads;
    @Option(longName = "xgb-seed", usage = "Sets the random seed for XGBoost.")
    private long seed = Trainer.DEFAULT_SEED;

    /**
     * Gets the configured XGBoostRegressionTrainer.
     *
     * @return The configured trainer.
     */
    public XGBoostRegressionTrainer getTrainer() {
        return new XGBoostRegressionTrainer(rType, ensembleSize, eta, gamma, depth, minWeight, subsample, subsampleFeatures, lambda, alpha, numThreads, quiet, seed);
    }
}
