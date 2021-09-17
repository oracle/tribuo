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

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.data.DataOptions;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.xgboost.XGBoostRegressionTrainer.RegressionType;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run an XGBoost regressor for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Command line options.
     */
    public static class XGBoostOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests an XGBoost regression model on the specified datasets.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;

        /**
         * Regression type to use. Defaults to LINEAR.
         */
        @Option(longName = "regression-metric", usage = "Regression type to use. Defaults to LINEAR.")
        public RegressionType rType = RegressionType.LINEAR;
        /**
         * Number of trees in the ensemble.
         */
        @Option(charName = 'm', longName = "ensemble-size", usage = "Number of trees in the ensemble.")
        public int ensembleSize = -1;
        /**
         * L1 regularization term for weights (default 0).
         */
        @Option(charName = 'a', longName = "alpha", usage = "L1 regularization term for weights (default 0).")
        public float alpha = 0.0f;
        /**
         * Minimum sum of instance weights needed in a leaf (default 1, range [0,inf]).
         */
        @Option(longName = "min-weight", usage = "Minimum sum of instance weights needed in a leaf (default 1, range [0,inf]).")
        public float minWeight = 1;
        /**
         * Max tree depth (default 6, range (0,inf]).
         */
        @Option(charName = 'd', longName = "max-depth", usage = "Max tree depth (default 6, range (0,inf]).")
        public int depth = 6;
        /**
         * Step size shrinkage parameter (default 0.3, range [0,1]).
         */
        @Option(charName = 'e', longName = "eta", usage = "Step size shrinkage parameter (default 0.3, range [0,1]).")
        public float eta = 0.3f;
        /**
         * Subsample features for each tree (default 1, range (0,1]).
         */
        @Option(longName = "subsample-features", usage = "Subsample features for each tree (default 1, range (0,1]).")
        public float subsampleFeatures = 1.0f;
        /**
         * Minimum loss reduction to make a split (default 0, range [0,inf]).
         */
        @Option(charName = 'g', longName = "gamma", usage = "Minimum loss reduction to make a split (default 0, range [0,inf]).")
        public float gamma = 0.0f;
        /**
         * L2 regularization term for weights (default 1).
         */
        @Option(charName = 'l', longName = "lambda", usage = "L2 regularization term for weights (default 1).")
        public float lambda = 1.0f;
        /**
         * Make the XGBoost training procedure quiet.
         */
        @Option(charName = 'q', longName = "quiet", usage = "Make the XGBoost training procedure quiet.")
        public boolean quiet;
        /**
         * Subsample size for each tree (default 1, range (0,1]).
         */
        @Option(longName = "subsample", usage = "Subsample size for each tree (default 1, range (0,1]).")
        public float subsample = 1.0f;
        /**
         * Number of threads to use (default 4, range (1, num hw threads)).
         */
        @Option(charName = 't', longName = "num-threads", usage = "Number of threads to use (default 4, range (1, num hw threads)).")
        public int numThreads = 4;
    }

    /**
     * Runs a TrainTest CLI.
     * @param args the command line arguments
     * @throws IOException if there is any error reading the examples.
     */
    public static void main(String[] args) throws IOException {
        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        XGBoostOptions o = new XGBoostOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.general.trainingPath == null || o.general.testingPath == null) {
            logger.info(cm.usage());
            logger.info("Please supply a training path and a testing path");
            return;
        }

        if (o.ensembleSize == -1) {
            logger.info(cm.usage());
            logger.info("Please supply the number of trees.");
            return;
        }

        RegressionFactory factory = new RegressionFactory();

        Pair<Dataset<Regressor>,Dataset<Regressor>> data = o.general.load(factory);
        Dataset<Regressor> train = data.getA();
        Dataset<Regressor> test = data.getB();

        //public XGBoostRegressionTrainer(RegressionType rType, int numTrees, double eta, double gamma, int maxDepth, double minChildWeight, double subsample, double featureSubsample, double lambda, double alpha, long seed) {
        XGBoostRegressionTrainer trainer = new XGBoostRegressionTrainer(o.rType,o.ensembleSize,o.eta,o.gamma,o.depth,o.minWeight,o.subsample,o.subsampleFeatures,o.lambda,o.alpha,o.numThreads,o.quiet,o.general.seed);
        logger.info("Training using " + trainer.toString());
        final long trainStart = System.currentTimeMillis();
        Model<Regressor> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();

        logger.info("Finished training regressor " + Util.formatDuration(trainStart,trainStop));

        final long testStart = System.currentTimeMillis();
        RegressionEvaluation evaluation = factory.getEvaluator().evaluate(model,test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart,testStop));
        System.out.println(evaluation.toString());

        if (o.general.outputPath != null) {
            o.general.saveModel(model);
        }
    }
}
