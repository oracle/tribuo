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

package org.tribuo.data;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.dataset.MinimumCardinalityDataset;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.transform.TransformTrainer;
import org.tribuo.transform.TransformationMap;
import org.tribuo.util.Util;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Build and run a predictor for a standard dataset.
 */
public final class CompletelyConfigurableTrainTest {

    private static final Logger logger = Logger.getLogger(CompletelyConfigurableTrainTest.class.getName());

    private CompletelyConfigurableTrainTest() {}

    /**
     * Command line options.
     */
    public static class ConfigurableTrainTestOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Loads a Trainer and two DataSources from a config file, trains a Model, tests it and optionally saves it to disk.";
        }

        /**
         * Path to serialize model to.
         */
        @Option(charName = 'f', longName = "model-output-path", usage = "Path to serialize model to.")
        public Path outputPath;

        /**
         * Load the training DataSource from the config file.
         */
        @Option(charName = 'u', longName = "train-source", usage = "Load the training DataSource from the config file.")
        public ConfigurableDataSource<?> trainSource;

        /**
         * Load the testing DataSource from the config file.
         */
        @Option(charName = 'v', longName = "test-source", usage = "Load the testing DataSource from the config file.")
        public ConfigurableDataSource<?> testSource;

        /**
         * Load a trainer from the config file.
         */
        @Option(charName = 't', longName = "trainer", usage = "Load a trainer from the config file.")
        public Trainer<?> trainer;

        /**
         * Load a transformation map from the config file.
         */
        @Option(longName = "transformer", usage = "Load a transformation map from the config file.")
        public TransformationMap transformationMap;

        /**
         * Remove features which occur fewer than m times.
         */
        @Option(charName = 'm', longName = "minimum-count", usage = "Remove features which occur fewer than <int> times.")
        public int minCount = -1;
    }

    /**
     * @param args the command line arguments
     * @param <T> The {@link Output} subclass.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Output<T>> void main(String[] args) {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        ConfigurableTrainTestOptions o = new ConfigurableTrainTestOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.trainSource == null || o.testSource == null) {
            logger.info(cm.usage());
            System.exit(1);
        } else if (o.trainer == null) {
            logger.warning("No trainer supplied");
            logger.info(cm.usage());
            System.exit(1);
        }

        Dataset<T> train = new MutableDataset<>((DataSource<T>)o.trainSource);
        if (o.minCount > 0) {
            logger.info("Removing features which occur fewer than " + o.minCount + " times.");
            train = new MinimumCardinalityDataset<>(train,o.minCount);
        }
        Dataset<T> test = new MutableDataset<>((DataSource<T>)o.testSource);

        if (o.transformationMap != null) {
            o.trainer = new TransformTrainer<>(o.trainer,o.transformationMap);
        }
        logger.info("Trainer is " + o.trainer.getProvenance().toString());

        logger.info("Outputs are " + train.getOutputInfo().toReadableString());

        logger.info("Number of features: " + train.getFeatureMap().size());

        final long trainStart = System.currentTimeMillis();
        Model<T> model = ((Trainer<T>)o.trainer).train(train);
        final long trainStop = System.currentTimeMillis();
                
        logger.info("Finished training classifier " + Util.formatDuration(trainStart,trainStop));

        Evaluator<T,? extends Evaluation<T>> evaluator = train.getOutputFactory().getEvaluator();
        final long testStart = System.currentTimeMillis();
        Evaluation<T> evaluation = evaluator.evaluate(model,test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart,testStop));
        System.out.println(evaluation.toString());

        if (o.outputPath != null) {
            try (ObjectOutputStream oout = new ObjectOutputStream(new FileOutputStream(o.outputPath.toFile()))) {
                oout.writeObject(model);
                logger.info("Serialized model to file: " + o.outputPath);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Error writing model", e);
            }
        }
    }
}
