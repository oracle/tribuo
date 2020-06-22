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

/**
 * CLI options for training an XGBoost classifier.
 */
public class XGBoostOptions implements ClassificationOptions<XGBoostClassificationTrainer> {
    @Option(longName = "xgb-ensemble-size", usage = "Number of trees in the ensemble.")
    public int xgbEnsembleSize = -1;
    @Option(longName = "xgb-alpha", usage = "L1 regularization term for weights (default 0).")
    public float xbgAlpha = 0.0f;
    @Option(longName = "xgb-min-weight", usage = "Minimum sum of instance weights needed in a leaf (default 1, range [0,inf]).")
    public float xgbMinWeight = 1;
    @Option(longName = "xgb-max-depth", usage = "Max tree depth (default 6, range (0,inf]).")
    public int xgbMaxDepth = 6;
    @Option(longName = "xgb-eta", usage = "Step size shrinkage parameter (default 0.3, range [0,1]).")
    public float xgbEta = 0.3f;
    @Option(longName = "xgb-subsample-features", usage = "Subsample features for each tree (default 1, range (0,1]).")
    public float xgbSubsampleFeatures;
    @Option(longName = "xgb-gamma", usage = "Minimum loss reduction to make a split (default 0, range [0,inf]).")
    public float xgbGamma = 0.0f;
    @Option(longName = "xgb-lambda", usage = "L2 regularization term for weights (default 1).")
    public float xgbLambda = 1.0f;
    @Option(longName = "xgb-quiet", usage = "Make the XGBoost training procedure quiet.")
    public boolean xgbQuiet;
    @Option(longName = "xgb-subsample", usage = "Subsample size for each tree (default 1, range (0,1]).")
    public float xgbSubsample = 1.0f;
    @Option(longName = "xgb-num-threads", usage = "Number of threads to use (default 4, range (1, num hw threads)).")
    public int xgbNumThreads;
    @Option(longName = "xgb-seed", usage = "Sets the random seed for XGBoost.")
    private long xgbSeed = Trainer.DEFAULT_SEED;

    @Override
    public XGBoostClassificationTrainer getTrainer() {
        if (xgbEnsembleSize == -1) {
            throw new IllegalArgumentException("Please supply the number of trees.");
        }
        return new XGBoostClassificationTrainer(xgbEnsembleSize, xgbEta, xgbGamma, xgbMaxDepth, xgbMinWeight, xgbSubsample, xgbSubsampleFeatures, xgbLambda, xbgAlpha, xgbNumThreads, xgbQuiet, xgbSeed);
    }
}
