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

package org.tribuo.classification.xgboost;

import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.Trainer;
import org.tribuo.classification.ClassificationOptions;
import org.tribuo.common.xgboost.XGBoostTrainer;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * CLI options for training an XGBoost classifier.
 */
public class XGBoostOptions implements ClassificationOptions<XGBoostClassificationTrainer> {
    private static final Logger logger = Logger.getLogger(XGBoostOptions.class.getName());

    /**
     * Weak learning algorithm.
     */
    @Option(longName = "xgb-booster-type", usage = "Weak learning algorithm.")
    public XGBoostTrainer.BoosterType xgbBoosterType = XGBoostTrainer.BoosterType.GBTREE;
    /**
     * Tree building algorithm.
     */
    @Option(longName = "xgb-tree-method", usage = "Tree building algorithm.")
    public XGBoostTrainer.TreeMethod xgbTreeMethod = XGBoostTrainer.TreeMethod.AUTO;
    /**
     * Number of trees in the ensemble.
     */
    @Option(longName = "xgb-ensemble-size", usage = "Number of trees in the ensemble.")
    public int xgbEnsembleSize = -1;
    /**
     * L1 regularization term for weights.
     */
    @Option(longName = "xgb-alpha", usage = "L1 regularization term for weights.")
    public float xbgAlpha = 0.0f;
    /**
     * Minimum sum of instance weights needed in a leaf (range [0,Infinity]).
     */
    @Option(longName = "xgb-min-weight", usage = "Minimum sum of instance weights needed in a leaf (range [0,Infinity]).")
    public float xgbMinWeight = 1;
    /**
     * Max tree depth (range (0,Integer.MAX_VALUE]).
     */
    @Option(longName = "xgb-max-depth", usage = "Max tree depth (range (0,Integer.MAX_VALUE]).")
    public int xgbMaxDepth = 6;
    /**
     * Step size shrinkage parameter (range [0,1]).
     */
    @Option(longName = "xgb-eta", usage = "Step size shrinkage parameter (range [0,1]).")
    public float xgbEta = 0.3f;
    /**
     * Subsample features for each tree (range (0,1]).
     */
    @Option(longName = "xgb-subsample-features", usage = "Subsample features for each tree (range (0,1]).")
    public float xgbSubsampleFeatures = 0.0f;
    /**
     * Minimum loss reduction to make a split (range [0,Infinity]).
     */
    @Option(longName = "xgb-gamma", usage = "Minimum loss reduction to make a split (range [0,Infinity]).")
    public float xgbGamma = 0.0f;
    /**
     * L2 regularization term for weights.
     */
    @Option(longName = "xgb-lambda", usage = "L2 regularization term for weights.")
    public float xgbLambda = 1.0f;
    /**
     * Deprecated, use xgb-loglevel.
     */
    @Option(longName = "xgb-quiet", usage = "Deprecated, use xgb-loglevel.")
    public boolean xgbQuiet;
    /**
     * Make the XGBoost training procedure quiet.
     */
    @Option(longName = "xgb-loglevel", usage = "Make the XGBoost training procedure quiet.")
    public XGBoostTrainer.LoggingVerbosity xgbLogLevel = XGBoostTrainer.LoggingVerbosity.WARNING;
    /**
     * Subsample size for each tree (range (0,1]).
     */
    @Option(longName = "xgb-subsample", usage = "Subsample size for each tree (range (0,1]).")
    public float xgbSubsample = 1.0f;
    /**
     * Number of threads to use (range (1, num hw threads)). The default of 0 means use all hw threads.
     */
    @Option(longName = "xgb-num-threads", usage = "Number of threads to use (range (1, num hw threads)). The default of 0 means use all hw threads.")
    public int xgbNumThreads = 0;
    @Option(longName = "xgb-seed", usage = "Sets the random seed for XGBoost.")
    private long xgbSeed = Trainer.DEFAULT_SEED;

    @Override
    public XGBoostClassificationTrainer getTrainer() {
        if (xgbEnsembleSize == -1) {
            throw new IllegalArgumentException("Please supply the number of trees.");
        }
        if (xgbQuiet) {
            logger.log(Level.WARNING, "Silencing XGBoost, overriding logging verbosity. Please switch to the 'xgb-loglevel' argument.");
            xgbLogLevel = XGBoostTrainer.LoggingVerbosity.SILENT;
        }
        return new XGBoostClassificationTrainer(xgbBoosterType, xgbTreeMethod, xgbEnsembleSize, xgbEta, xgbGamma, xgbMaxDepth, xgbMinWeight, xgbSubsample, xgbSubsampleFeatures, xgbLambda, xbgAlpha, xgbNumThreads, xgbLogLevel, xgbSeed);
    }
}
