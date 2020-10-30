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

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.data.DataOptions;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.libsvm.SVMRegressionType.SVMMode;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Build and run a LibSVM regressor for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    public static class LibLinearOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a LibSVM regression model on the specified datasets.";
        }
        public DataOptions general;

        @Option(longName="coefficient",usage="Intercept in kernel function.")
        public double coeff = 1.0;
        @Option(charName='d',longName="degree",usage="Degree in polynomial kernel.")
        public int degree = 3;
        @Option(charName='g',longName="gamma",usage="Gamma value in kernel function.")
        public double gamma = 0.0;
        @Option(charName='k',longName="kernel",usage="Type of SVM kernel.")
        public KernelType kernelType = KernelType.LINEAR;
        @Option(charName='t',longName="type",usage="Type of SVM.")
        public SVMRegressionType.SVMMode svmType = SVMMode.EPSILON_SVR;
        @Option(longName="standardize",usage="Standardize the regression outputs internally to the SVM")
        public boolean standardize = false;
    }

    /**
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

        SVMParameters<Regressor> parameters = new SVMParameters<>(new SVMRegressionType(o.svmType), o.kernelType);
        parameters.setGamma(o.gamma);
        parameters.setCoeff(o.coeff);
        parameters.setDegree(o.degree);
        Trainer<Regressor> trainer = new LibSVMRegressionTrainer(parameters, o.standardize);
        logger.info("Training using " + trainer.toString());

        final long trainStart = System.currentTimeMillis();
        Model<Regressor> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();

        logger.info("Finished training regressor " + Util.formatDuration(trainStart,trainStop));
        logger.info("Support vectors - " + ((LibSVMRegressionModel)model).getNumberOfSupportVectors());

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
