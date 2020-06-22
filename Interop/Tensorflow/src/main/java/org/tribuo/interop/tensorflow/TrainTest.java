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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Build and run a Tensorflow multi-class classifier for a standard dataset.
 */
public class TrainTest {

    private static final Logger logger = Logger.getLogger(TrainTest.class.getName());

    public enum InputType { DENSE, IMAGE }

    public static Pair<Dataset<Label>,Dataset<Label>> load(Path trainingPath, Path testingPath, OutputFactory<Label> outputFactory) throws IOException {
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

    public static void saveModel(Path outputPath, Model<Label> model) throws IOException {
        FileOutputStream fout = new FileOutputStream(outputPath.toFile());
        ObjectOutputStream oout = new ObjectOutputStream(fout);
        oout.writeObject(model);
        oout.close();
        fout.close();
        logger.info("Serialized model to file: " + outputPath);
    }

    public static class TensorflowOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Trains and tests a Tensorflow model.";
        }
        @Option(charName='f',longName="model-output-path",usage="Path to serialize model to.")
        public Path outputPath;
        @Option(charName='u',longName="training-file",usage="Path to the libsvm format training file.")
        public Path trainingPath;
        @Option(charName='v',longName="testing-file",usage="Path to the libsvm format testing file.")
        public Path testingPath;

        @Option(charName='b',longName="batch-size",usage="Test time minibatch size.")
        public int testBatchSize = 16;

        @Option(charName='b',longName="batch-size",usage="Minibatch size.")
        public int batchSize = 128;
        @Option(charName='e',longName="num-epochs",usage="Number of gradient descent epochs.")
        public int epochs = 5;
        @Option(charName='i',longName="image-format",usage="Image format, in [W,H,C]. Defaults to MNIST.")
        public String imageFormat = "28,28,1";
        @Option(charName='t',longName="input-type",usage="Input type.")
        public InputType inputType = InputType.IMAGE;
        @Option(charName='m',longName="model-protobuf",usage="Path to the protobuf containing the network description.")
        public Path protobufPath;
        @Option(charName='p',longName="checkpoint-dir",usage="Path to the checkpoint base directory.")
        public Path checkpointPath;
    }

    /**
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

        ExampleTransformer<Label> inputTransformer;
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
                inputTransformer = new ImageTransformer<>(width,height,channels);
                break;
            case DENSE:
                inputTransformer = new DenseTransformer<>();
                break;
            default:
                logger.info(cm.usage());
                logger.info("Unknown input type. Found " + o.inputType);
                return;
        }
        OutputTransformer<Label> labelTransformer = new LabelTransformer();

        //public TensorflowTrainer(Path graphPath, ExampleTransformer<T> exampleTransformer, OutputTransformer<T> outputTransformer, int batchSize, int numEpochs) throws IOException {
        Trainer<Label> trainer;
        if (o.checkpointPath == null) {
            logger.info("Using TensorflowTrainer");
            trainer = new TensorflowTrainer<>(o.protobufPath, inputTransformer, labelTransformer, o.batchSize, o.epochs, o.testBatchSize);
        } else {
            logger.info("Using TensorflowCheckpointTrainer, writing to path " + o.checkpointPath);
            trainer = new TensorflowCheckpointTrainer<>(o.protobufPath, o.checkpointPath, inputTransformer, labelTransformer, o.batchSize, o.epochs);
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

        logger.info(evaluation.toString());

        logger.info(evaluation.getConfusionMatrix().toString());

        if (o.outputPath != null) {
            saveModel(o.outputPath, model);
        }

        if (o.checkpointPath == null) {
            ((TensorflowModel<?>) model).close();
        } else {
            ((TensorflowCheckpointModel<?>) model).close();
        }
    }
}
