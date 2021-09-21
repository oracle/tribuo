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

package org.tribuo.regression.slm;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.data.DataOptions;
import org.tribuo.math.la.SparseVector;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Build and run a sparse linear regression model for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Type of sparse linear model.
     */
    public enum SLMType {
        /**
         * Creates a {@link SLMTrainer} which performs sequential forward selection of the features.
         */
        SFS,
        /**
         * Creates a {@link SLMTrainer} which performs sequential forward selection on normalised features.
         */
        SFSN,
        /**
         * Creates a {@link LARSTrainer}.
         */
        LARS,
        /**
         * Creates a {@link LARSLassoTrainer}.
         */
        LARSLASSO,
        /**
         * Creates an {@link ElasticNetCDTrainer}.
         */
        ELASTICNET
    }

    /**
     * Command line options.
     */
    public static class SLMOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a sparse linear regression model on the specified datasets.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;

        /**
         * Set the maximum number of features.
         */
        @Option(charName = 'm', longName = "max-features-num", usage = "Set the maximum number of features.")
        public int maxNumFeatures = -1;
        /**
         * Choose the training algorithm (stepwise forward selection or least angle regression).
         */
        @Option(charName = 'a', longName = "algorithm", usage = "Choose the training algorithm (stepwise forward selection or least angle regression).")
        public SLMType algorithm = SLMType.LARS;
        /**
         * Regularisation strength in the Elastic Net.
         */
        @Option(charName = 'b', longName = "alpha", usage = "Regularisation strength in the Elastic Net.")
        public double alpha = 1.0;
        /**
         * Ratio between the l1 and l2 penalties in the Elastic Net. Must be between 0 and 1.
         */
        @Option(charName = 'l', longName = "l1Ratio", usage = "Ratio between the l1 and l2 penalties in the Elastic Net. Must be between 0 and 1.")
        public double l1Ratio = 1.0;
        /**
         * Iterations of Elastic Net.
         */
        @Option(longName = "iterations", usage = "Iterations of Elastic Net.")
        public int iterations = 500;
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

        SLMOptions o = new SLMOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.general.trainingPath == null || o.general.testingPath == null) {
            logger.info(cm.usage());
            return;
        }

        RegressionFactory factory = new RegressionFactory();

        Pair<Dataset<Regressor>,Dataset<Regressor>> data = o.general.load(factory);
        Dataset<Regressor> train = data.getA();
        Dataset<Regressor> test = data.getB();

        SparseTrainer<Regressor> trainer;

        switch (o.algorithm) {
            case SFS:
                trainer = new SLMTrainer(false,Math.min(train.getFeatureMap().size(),o.maxNumFeatures));
                break;
            case LARS:
                trainer = new LARSTrainer(Math.min(train.getFeatureMap().size(),o.maxNumFeatures));
                break;
            case LARSLASSO:
                trainer = new LARSLassoTrainer(Math.min(train.getFeatureMap().size(),o.maxNumFeatures));
                break;
            case SFSN:
                trainer = new SLMTrainer(true,Math.min(train.getFeatureMap().size(),o.maxNumFeatures));
                break;
            case ELASTICNET:
                trainer = new ElasticNetCDTrainer(o.alpha,o.l1Ratio,1e-4,o.iterations,false,o.general.seed);
                break;
            default:
                logger.warning("Unknown SLMType, found " + o.algorithm);
                return;
        }

        logger.info("Training using " + trainer.toString());
        final long trainStart = System.currentTimeMillis();
        SparseModel<Regressor> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();
        logger.info("Finished training regressor " + Util.formatDuration(trainStart,trainStop));
        logger.info("Selected features: " + model.getActiveFeatures());
        Map<String,SparseVector> weights = ((SparseLinearModel)model).getWeights();
        for (Map.Entry<String,SparseVector> e : weights.entrySet()) {
            logger.info("Target:" + e.getKey());
            logger.info("\tWeights: " + e.getValue());
            logger.info("\tWeights one norm: " + e.getValue().oneNorm());
            logger.info("\tWeights two norm: " + e.getValue().twoNorm());
        }
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
