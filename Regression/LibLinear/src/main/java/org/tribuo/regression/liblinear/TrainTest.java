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

package org.tribuo.regression.liblinear;

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
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.liblinear.LinearRegressionType.LinearType;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a LibLinear regressor for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Command line options.
     */
    public static class LibLinearOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a LibLinear regression model on the specified datasets.";
        }

        /**
         * The data loading options.
         */
        public DataOptions general;

        /**
         * Cost penalty for SVM.
         */
        @Option(charName = 'p', longName = "cost-penalty", usage = "Cost penalty for SVM.")
        public double cost = 1.0;
        /**
         * Max iterations over the data.
         */
        @Option(longName = "max-iterations", usage = "Max iterations over the data.")
        public int maxIterations = 1000;
        /**
         * Regression value insensitivity for margin.
         */
        @Option(longName = "epsilon-insensitivity", usage = "Regression value insensitivity for margin.")
        public double epsilon = 0.1;
        /**
         * Tolerance of the optimization termination criterion.
         */
        @Option(charName = 'e', longName = "termination-criterion", usage = "Tolerance of the optimization termination criterion.")
        public double terminationCriterion = 0.01;
        /**
         * Type of SVR.
         */
        @Option(charName = 't', longName = "algorithm", usage = "Type of SVR.")
        public LinearType algorithm = LinearType.L2R_L2LOSS_SVR;
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

        LibLinearOptions o = new LibLinearOptions();
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

        Trainer<Regressor> trainer = new LibLinearRegressionTrainer(new LinearRegressionType(o.algorithm),o.cost,o.maxIterations,o.terminationCriterion,o.epsilon);
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
