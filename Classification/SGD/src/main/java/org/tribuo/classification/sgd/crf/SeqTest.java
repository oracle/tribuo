/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.crf;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import org.tribuo.classification.Label;
import org.tribuo.classification.sequence.LabelSequenceEvaluation;
import org.tribuo.classification.sequence.LabelSequenceEvaluator;
import org.tribuo.classification.sequence.example.SequenceDataGenerator;
import org.tribuo.hash.HashCodeHasher;
import org.tribuo.hash.HashingOptions.ModelHashingType;
import org.tribuo.hash.MessageDigestHasher;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.math.optimisers.GradientOptimiserOptions;
import org.tribuo.sequence.HashingSequenceTrainer;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceTrainer;
import org.tribuo.util.Util;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Build and run a sequence classifier on a generated dataset.
 */
public class SeqTest {

    private static final Logger logger = Logger.getLogger(SeqTest.class.getName());

    /**
     * Command line options.
     */
    public static class CRFOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Tests a linear chain CRF model on the specified dataset.";
        }

        /**
         * Gradient descent options.
         */
        public GradientOptimiserOptions gradientOptions;
        /**
         * Name of the example dataset, options are {gorilla}.
         */
        @Option(charName = 'd', longName = "dataset-name", usage = "Name of the example dataset, options are {gorilla}.")
        public String datasetName = "";
        /**
         * Path to serialize model to.
         */
        @Option(charName = 'f', longName = "output-path", usage = "Path to serialize model to.")
        public Path outputPath;
        /**
         * Number of SGD epochs.
         */
        @Option(charName = 'i', longName = "epochs", usage = "Number of SGD epochs.")
        public int epochs = 5;
        /**
         * Print out feature, label and other model details.
         */
        @Option(charName = 'o', longName = "print-model", usage = "Print out feature, label and other model details.")
        public boolean logModel = false;
        /**
         * Log the objective after n examples.
         */
        @Option(charName = 'p', longName = "logging-interval", usage = "Log the objective after <int> examples.")
        public int loggingInterval = 100;
        /**
         * RNG seed.
         */
        @Option(charName = 'r', longName = "seed", usage = "RNG seed.")
        public long seed = 1;
        /**
         * Shuffle the data each epoch (default: true).
         */
        @Option(longName = "shuffle", usage = "Shuffle the data each epoch (default: true).")
        public boolean shuffle = true;
        /**
         * Path to a serialised SequenceDataset used for training.
         */
        @Option(charName = 'u', longName = "train-dataset", usage = "Path to a serialised SequenceDataset used for training.")
        public Path trainDataset = null;
        /**
         * Path to a serialised SequenceDataset used for testing.
         */
        @Option(charName = 'v', longName = "test-dataset", usage = "Path to a serialised SequenceDataset used for testing.")
        public Path testDataset = null;
        /**
         * Hash the model during training. Defaults to no hashing.
         */
        @Option(longName = "model-hashing-algorithm", usage = "Hash the model during training. Defaults to no hashing.")
        public ModelHashingType modelHashingAlgorithm = ModelHashingType.NONE;
        /**
         * Salt for hashing the model.
         */
        @Option(longName = "model-hashing-salt", usage = "Salt for hashing the model.")
        public String modelHashingSalt = "";
        /**
         * Load in the model in protobuf format.
         */
        @Option(longName = "protobuf-model", usage = "Load the model from a protobuf. Optional")
        public boolean protobufFormat;
    }

    /**
     * @param args the command line arguments
     * @throws ClassNotFoundException if it failed to load the model.
     * @throws IOException            if there is any error reading the examples.
     */
    @SuppressWarnings("unchecked") // deserialising a generic dataset.
    public static void main(String[] args) throws ClassNotFoundException, IOException {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        CRFOptions o = new CRFOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args, o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        logger.info("Configuring gradient optimiser");
        StochasticGradientOptimiser grad = o.gradientOptions.getOptimiser();

        logger.info(String.format("Set logging interval to %d", o.loggingInterval));

        SequenceDataset<Label> train;
        SequenceDataset<Label> test;
        switch (o.datasetName) {
            case "Gorilla":
            case "gorilla":
                logger.info("Generating gorilla dataset");
                train = SequenceDataGenerator.generateGorillaDataset(1);
                test = SequenceDataGenerator.generateGorillaDataset(1);
                break;
            default:
                if ((o.trainDataset != null) && (o.testDataset != null)) {
                    if (o.protobufFormat) {
                        logger.info("Loading protobuf format training data from " + o.trainDataset);
                        SequenceDataset<?> tmpTrain = SequenceDataset.deserializeFromFile(o.trainDataset);
                        train = SequenceDataset.castDataset(tmpTrain, Label.class);
                        logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                        logger.info("Found " + train.getFeatureIDMap().size() + " features");
                        logger.info("Loading protobuf format testing data from " + o.testDataset);
                        SequenceDataset<?> tmpTest = SequenceDataset.deserializeFromFile(o.testDataset);
                        test = SequenceDataset.castDataset(tmpTest, Label.class);
                        logger.info(String.format("Loaded %d testing examples", test.size()));
                    } else {
                        logger.info("Loading training data from " + o.trainDataset);
                        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(Files.newInputStream(o.trainDataset)));
                             ObjectInputStream oits = new ObjectInputStream(new BufferedInputStream(Files.newInputStream(o.testDataset)))) {
                            @SuppressWarnings("unchecked") // deserialising a generic dataset.
                            SequenceDataset<Label> tmpTrain = (SequenceDataset<Label>) ois.readObject();
                            train = tmpTrain;
                            logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                            logger.info("Found " + train.getFeatureIDMap().size() + " features");
                            logger.info("Loading testing data from " + o.testDataset);
                            @SuppressWarnings("unchecked") // deserialising a generic dataset.
                            SequenceDataset<Label> tmpTest = (SequenceDataset<Label>) oits.readObject();
                            test = tmpTest;
                            logger.info(String.format("Loaded %d testing examples", test.size()));
                        }
                    }
                } else {
                    logger.warning("Unknown dataset " + o.datasetName);
                    logger.info(cm.usage());
                    return;
                }
        }

        SequenceTrainer<Label> trainer = new CRFTrainer(grad, o.epochs, o.loggingInterval, o.seed);
        ((CRFTrainer) trainer).setShuffle(o.shuffle);
        switch (o.modelHashingAlgorithm) {
            case NONE:
                break;
            case HC:
                trainer = new HashingSequenceTrainer<>(trainer, new HashCodeHasher(o.modelHashingSalt));
                break;
            case SHA1:
                trainer = new HashingSequenceTrainer<>(trainer, new MessageDigestHasher("SHA1", o.modelHashingSalt));
                break;
            case SHA256:
                trainer = new HashingSequenceTrainer<>(trainer, new MessageDigestHasher("SHA-256", o.modelHashingSalt));
                break;
            default:
                logger.info("Unknown hasher " + o.modelHashingAlgorithm);
        }

        logger.info("Training using " + trainer.toString());
        final long trainStart = System.currentTimeMillis();
        CRFModel model = (CRFModel) trainer.train(train);
        final long trainStop = System.currentTimeMillis();
        logger.info("Finished training classifier " + Util.formatDuration(trainStart, trainStop));

        if (o.logModel) {
            System.out.println("FeatureMap = " + model.getFeatureIDMap().toString());
            System.out.println("LabelMap = " + model.getOutputIDInfo().toString());
            System.out.println("Features - " + model.generateWeightsString());
        }

        LabelSequenceEvaluator labelEvaluator = new LabelSequenceEvaluator();
        final long testStart = System.currentTimeMillis();
        LabelSequenceEvaluation evaluation = labelEvaluator.evaluate(model,test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart, testStop));
        System.out.println(evaluation.toString());
        System.out.println();
        System.out.println(evaluation.getConfusionMatrix().toString());

        if (o.outputPath != null) {
            if (o.protobufFormat) {
                model.serializeToFile(o.outputPath);
            } else {
                try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(o.outputPath))) {
                    oos.writeObject(model);
                }
            }
            logger.info("Serialized model to file: " + o.outputPath);
        }
    }
}
