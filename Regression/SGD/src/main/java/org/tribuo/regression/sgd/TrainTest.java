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

package org.tribuo.regression.sgd;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.data.DataOptions;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.optimisers.GradientOptimiserOptions;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.sgd.linear.LinearSGDTrainer;
import org.tribuo.regression.sgd.objectives.AbsoluteLoss;
import org.tribuo.regression.sgd.objectives.Huber;
import org.tribuo.regression.sgd.objectives.SquaredLoss;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a linear regression for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Loss function.
     */
    public enum LossEnum {
        /**
         * Uses {@link AbsoluteLoss}.
         */
        ABSOLUTE,
        /**
         * Uses {@link SquaredLoss}.
         */
        SQUARED,
        /**
         * Uses {@link Huber} loss.
         */
        HUBER
    }

    /**
     * Command line options.
     */
    public static class SGDOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a linear SGD regression model on the specified datasets.";
        }
        public DataOptions general;
        public GradientOptimiserOptions gradientOptions;

        @Option(charName='i',longName="epochs",usage="Number of SGD epochs. Defaults to 5.")
        public int epochs = 5;
        @Option(charName='o',longName="objective",usage="Loss function. Defaults to SQUARED.")
        public LossEnum loss = LossEnum.SQUARED;
        @Option(charName='p',longName="logging-interval",usage="Log the objective after <int> examples. Defaults to 100.")
        public int loggingInterval = 100;
        @Option(charName='z',longName="minibatch-size",usage="Minibatch size. Defaults to 1.")
        public int minibatchSize = 1;
    }

    /**
     * @param args the command line arguments
     * @throws IOException if there is any error reading the examples.
     */
    public static void main(String[] args) throws IOException {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        SGDOptions o = new SGDOptions();
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

        logger.info("Configuring gradient optimiser");
        RegressionObjective obj = null;
        switch (o.loss) {
            case ABSOLUTE:
                obj = new AbsoluteLoss();
                break;
            case SQUARED:
                obj = new SquaredLoss();
                break;
            case HUBER:
                obj = new Huber();
                break;
            default:
                logger.warning("Unknown objective function " + o.loss);
                logger.info(cm.usage());
                return;
        }
        StochasticGradientOptimiser grad = o.gradientOptions.getOptimiser();

        logger.info(String.format("Set logging interval to %d",o.loggingInterval));
        RegressionFactory factory = new RegressionFactory();

        Pair<Dataset<Regressor>,Dataset<Regressor>> data = o.general.load(factory);
        Dataset<Regressor> train = data.getA();
        Dataset<Regressor> test = data.getB();

        Trainer<Regressor> trainer = new LinearSGDTrainer(obj,grad,o.epochs,o.loggingInterval,o.minibatchSize,o.general.seed);
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

        //System.out.println("Features - " + model.getTopFeatures(model.getFeatureIDMap().size()+1));

        if (o.general.outputPath != null) {
            o.general.saveModel(model);
        }
    }
}
