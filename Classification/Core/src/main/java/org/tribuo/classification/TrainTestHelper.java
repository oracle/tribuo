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

package org.tribuo.classification;

import com.oracle.labs.mlrg.olcut.config.ArgumentException;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.data.DataOptions;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * This class provides static methods used by the demo classes in each classification backend.
 */
public final class TrainTestHelper {

    private static final Logger logger = Logger.getLogger(TrainTestHelper.class.getName());

    private static final LabelFactory factory = new LabelFactory();

    private TrainTestHelper() { }

    /**
     * This method trains a model on the specified training data, and evaluates it
     * on the specified test data. It writes out the timing to it's logger, and the
     * statistical performance to standard out. If set, the model is written out
     * to the specified path on disk.
     * @param cm The configuration manager which knows the arguments.
     * @param dataOptions The data options which specify the training and test data.
     * @param trainer The trainer to use.
     * @return The trained model.
     * @throws IOException If the data failed to load.
     */
    public static Model<Label> run(ConfigurationManager cm, DataOptions dataOptions, Trainer<Label> trainer) throws IOException {
        LabsLogFormatter.setAllLogFormatters();

        if (dataOptions.trainingPath == null || dataOptions.testingPath == null) {
            logger.info(cm.usage());
            logger.info("Training Path = " + dataOptions.trainingPath + ", Testing Path = " + dataOptions.testingPath);
            throw new ArgumentException("training-file","test-file","Must supply both training and testing data.");
        }

        Pair<Dataset<Label>, Dataset<Label>> data = dataOptions.load(factory);
        Dataset<Label> train = data.getA();
        logger.info("Training data has " + train.getFeatureIDMap().size() + " features.");

        Dataset<Label> test = data.getB();

        logger.info("Training using " + trainer.toString());
        final long trainStart = System.currentTimeMillis();
        Model<Label> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();
        logger.info("Finished training classifier " + Util.formatDuration(trainStart, trainStop));
        final long testStart = System.currentTimeMillis();
        LabelEvaluation evaluation = factory.getEvaluator().evaluate(model, test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart, testStop));

        if (model.generatesProbabilities()) {
            logger.info("Average AUC = " + evaluation.averageAUCROC(false));
            logger.info("Average weighted AUC = " + evaluation.averageAUCROC(true));
        }

        System.out.println(evaluation.toString());

        System.out.println(evaluation.getConfusionMatrix().toString());

        if (dataOptions.outputPath != null) {
            dataOptions.saveModel(model);
        }

        return model;
    }
}
