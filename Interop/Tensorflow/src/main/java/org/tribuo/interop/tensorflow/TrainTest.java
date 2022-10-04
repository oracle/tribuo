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

package org.tribuo.interop.tensorflow;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.ImmutableDataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.datasource.LibSVMDataSource;
import org.tribuo.util.Util;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Build and run a Tensorflow multi-class classifier for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    /**
     * Type of feature extractor.
     */
    public enum InputType {
        /**
         * Dense feature extractor.
         */
        DENSE,
        /**
         * Image feature extractor, requires the image format option be set.
         */
        IMAGE
    }

    private static Pair<Dataset<Label>,Dataset<Label>> load(Path trainingPath, Path testingPath, OutputFactory<Label> outputFactory) throws IOException {
        logger.info(String.format("Loading data from %s", trainingPath));
        Dataset<Label> train;
        Dataset<Label> test;
        //
        // Load the libsvm text-based training data format.
        LibSVMDataSource<Label> trainSource = new LibSVMDataSource<>(trainingPath,outputFactory);
        train = new MutableDataset<>(trainSource);
        boolean zeroIndexed = trainSource.isZeroIndexed();
        int maxFeatureID = trainSource.getMaxFeatureID();
        logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
        logger.info("Found " + train.getFeatureIDMap().size() + " features");
        test = new ImmutableDataset<>(new LibSVMDataSource<>(testingPath,outputFactory,zeroIndexed,maxFeatureID),train.getFeatureIDMap(),train.getOutputIDInfo(),false);
        logger.info(String.format("Loaded %d testing examples", test.size()));
        return new Pair<>(train,test);
    }

    /**
     * Options for training a model in TensorFlow.
     */
    public static class TensorflowOptions implements Options {
        private static List<String> DEFAULT_PARAM_NAMES = new ArrayList<>();
        private static List<Float> DEFAULT_PARAM_VALUES = new ArrayList<>();

        static {
            DEFAULT_PARAM_NAMES.add("learningRate");
            DEFAULT_PARAM_NAMES.add("initialAccumulatorValue");
            DEFAULT_PARAM_VALUES.add(0.01f);
            DEFAULT_PARAM_VALUES.add(0.1f);
        }

        @Override
        public String getOptionsDescription() {
            return "Trains and tests a Tensorflow classification model.";
        }

        /**
         * Path to serialize model to.
         */
        @Option(charName = 'f', longName = "model-output-path", usage = "Path to serialize model to.")
        public Path outputPath;
        /**
         * Save the Tribuo model out as a protobuf.
         */
        @Option(longName = "model-save-to-proto", usage = "Save the Tribuo model out as a protobuf.")
        public boolean saveToProto;
        /**
         * Path to the libsvm format training file.
         */
        @Option(charName = 'u', longName = "training-file", usage = "Path to the libsvm format training file.")
        public Path trainingPath;
        /**
         * Path to the libsvm format testing file.
         */
        @Option(charName = 'v', longName = "testing-file", usage = "Path to the libsvm format testing file.")
        public Path testingPath;
        /**
         * Name of the output operation.
         */
        @Option(charName = 'l', longName = "output-name", usage = "Name of the output operation.")
        public String outputName;
        /**
         * Gradient optimizer param names, see org.tribuo.interop.tensorflow.GradientOptimiser.
         */
        @Option(longName = "optimizer-param-names", usage = "Gradient optimizer param names, see org.tribuo.interop.tensorflow.GradientOptimiser.")
        public List<String> gradientParamNames = DEFAULT_PARAM_NAMES;
        /**
         * Gradient optimizer param values, see org.tribuo.interop.tensorflow.GradientOptimiser.
         */
        @Option(longName = "optimizer-param-values", usage = "Gradient optimizer param values, see org.tribuo.interop.tensorflow.GradientOptimiser.")
        public List<Float> gradientParamValues = DEFAULT_PARAM_VALUES;
        /**
         * The gradient optimizer to use.
         */
        @Option(charName = 'g', longName = "gradient-optimizer", usage = "The gradient optimizer to use.")
        public GradientOptimiser optimiser = GradientOptimiser.ADAGRAD;
        /**
         * Test time minibatch size.
         */
        @Option(longName = "test-batch-size", usage = "Test time minibatch size.")
        public int testBatchSize = 16;
        /**
         * Minibatch size.
         */
        @Option(charName = 'b', longName = "batch-size", usage = "Minibatch size.")
        public int batchSize = 128;
        /**
         * Number of gradient descent epochs.
         */
        @Option(charName = 'e', longName = "num-epochs", usage = "Number of gradient descent epochs.")
        public int epochs = 5;
        /**
         * Interval between logging the loss.
         */
        @Option(longName = "logging-interval", usage = "Interval between logging the loss.")
        public int loggingInterval = 1000;
        /**
         * Name of the input placeholder.
         */
        @Option(charName = 'n', longName = "input-name", usage = "Name of the input placeholder.")
        public String inputName;
        /**
         * Image format, in [W,H,C]. Defaults to MNIST.
         */
        @Option(longName = "image-format", usage = "Image format, in [W,H,C]. Defaults to MNIST.")
        public String imageFormat = "28,28,1";
        /**
         * Input type.
         */
        @Option(charName = 't', longName = "input-type", usage = "Input type.")
        public InputType inputType = InputType.IMAGE;
        /**
         * Path to the protobuf containing the network description.
         */
        @Option(charName = 'm', longName = "model-protobuf", usage = "Path to the protobuf containing the network description.")
        public Path protobufPath;
        /**
         * Path to the checkpoint base directory.
         */
        @Option(charName = 'p', longName = "checkpoint-dir", usage = "Path to the checkpoint base directory.")
        public Path checkpointPath;

        /**
         * Zips the gradient parameter names and values.
         *
         * @return The gradient parameter map.
         */
        public Map<String, Float> getGradientParams() {
            if (gradientParamNames.size() != gradientParamValues.size()) {
                throw new IllegalArgumentException("Must supply both name and value for the gradient parameters, " +
                        "found " + gradientParamNames.size() + " names, and " + gradientParamValues.size() + "values.");
            }
            Map<String, Float> output = new HashMap<>();
            for (int i = 0; i < gradientParamNames.size(); i++) {
                output.put(gradientParamNames.get(i), gradientParamValues.get(i));
            }
            return output;
        }
    }

    /**
     * CLI entry point.
     * @param args the command line arguments
     * @throws IOException if there is any error reading the examples.
     */
    public static void main(String[] args) throws IOException {
        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        TensorflowOptions o = new TensorflowOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.trainingPath == null || o.testingPath == null) {
            logger.info(cm.usage());
            return;
        }

        Pair<Dataset<Label>,Dataset<Label>> data = load(o.trainingPath, o.testingPath, new LabelFactory());
        Dataset<Label> train = data.getA();
        Dataset<Label> test = data.getB();

        if ((o.inputName == null || o.inputName.isEmpty()) || (o.outputName == null || o.outputName.isEmpty())) {
            throw new IllegalArgumentException("Must specify both 'input-name' and 'output-name'");
        }

        FeatureConverter inputConverter;
        switch (o.inputType) {
            case IMAGE:
                String[] splitFormat = o.imageFormat.split(",");
                if (splitFormat.length != 3) {
                    logger.info(cm.usage());
                    logger.info("Invalid image format specified. Found " + o.imageFormat);
                    return;
                }
                int width = Integer.parseInt(splitFormat[0]);
                int height = Integer.parseInt(splitFormat[1]);
                int channels = Integer.parseInt(splitFormat[2]);
                inputConverter = new ImageConverter(o.inputName,width,height,channels);
                break;
            case DENSE:
                inputConverter = new DenseFeatureConverter(o.inputName);
                break;
            default:
                logger.info(cm.usage());
                logger.info("Unknown input type. Found " + o.inputType);
                return;
        }
        OutputConverter<Label> labelConverter = new LabelConverter();

        Trainer<Label> trainer;
        if (o.checkpointPath == null) {
            logger.info("Using TensorflowTrainer");
            trainer = new TensorFlowTrainer<>(o.protobufPath, o.outputName, o.optimiser, o.getGradientParams(), inputConverter, labelConverter, o.batchSize, o.epochs, o.testBatchSize, o.loggingInterval);
        } else {
            logger.info("Using TensorflowCheckpointTrainer, writing to path " + o.checkpointPath);
            trainer = new TensorFlowTrainer<>(o.protobufPath, o.outputName, o.optimiser, o.getGradientParams(), inputConverter, labelConverter, o.batchSize, o.epochs, o.testBatchSize, o.loggingInterval, o.checkpointPath);
        }
        logger.info("Training using " + trainer.toString());
        final long trainStart = System.currentTimeMillis();
        Model<Label> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();
        logger.info("Finished training classifier " + Util.formatDuration(trainStart, trainStop));
        final long testStart = System.currentTimeMillis();
        LabelEvaluator evaluator = new LabelEvaluator();
        LabelEvaluation evaluation = evaluator.evaluate(model, test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart, testStop));

        if (model.generatesProbabilities()) {
            logger.info("Average AUC = " + evaluation.averageAUCROC(false));
            logger.info("Average weighted AUC = " + evaluation.averageAUCROC(true));
        }

        System.out.println(evaluation.toString());

        System.out.println(evaluation.getConfusionMatrix().toString());

        if (o.outputPath != null) {
            if (o.saveToProto) {
                model.serializeToFile(o.outputPath);
            } else {
                try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(o.outputPath))) {
                    oos.writeObject(model);
                }
            }
            logger.info("Serialized model to file: " + o.outputPath);
        }

        if (o.checkpointPath == null) {
            ((TensorFlowNativeModel<?>) model).close();
        } else {
            ((TensorFlowCheckpointModel<?>) model).close();
        }
    }
}
