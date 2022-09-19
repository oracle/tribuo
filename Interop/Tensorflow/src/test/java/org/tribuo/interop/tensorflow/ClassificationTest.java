/*
 * Copyright (c) 2021, 2022 Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.IntNdArray;
import org.tensorflow.proto.framework.GraphDef;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ArrayExample;
import org.tribuo.interop.tensorflow.example.CNNExamples;
import org.tribuo.interop.tensorflow.example.GraphDefTuple;
import org.tribuo.interop.tensorflow.example.MLPExamples;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.test.Helpers;
import org.tribuo.util.Util;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ClassificationTest {

    private static final int PIXEL_DEPTH = 255;
    private static final int IMAGE_SIZE = 28;
    private static final int NUM_LABELS = 10;
    private static final String INPUT_NAME = "inputplaceholder";

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(TensorFlowTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void classificationMLPTest() throws IOException {
        // Create the train and test data
        Pair<Dataset<Label>, Dataset<Label>> data = LabelledDataGenerator.denseTrainTest(-1.0);
        Dataset<Label> trainData = data.getA();
        Dataset<Label> testData = data.getB();

        // Build the MLP graph
        GraphDefTuple graphDefTuple = MLPExamples.buildMLPGraph(INPUT_NAME, trainData.getFeatureMap().size(), new int[]{5, 5}, trainData.getOutputs().size());

        // Configure the trainer
        Map<String, Float> gradientParams = new HashMap<>();
        gradientParams.put("learningRate", 0.01f);
        gradientParams.put("initialAccumulatorValue", 0.1f);
        FeatureConverter denseConverter = new DenseFeatureConverter(INPUT_NAME);
        OutputConverter<Label> outputConverter = new LabelConverter();

        // Test native trainer
        TensorFlowTrainer<Label> nativeTrainer = new TensorFlowTrainer<>(graphDefTuple.graphDef,
                graphDefTuple.outputName,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                denseConverter,
                outputConverter,
                16,
                5,
                16,
                -1);

        @SuppressWarnings("unchecked")
        TensorFlowNativeModel<Label> nativeModel = (TensorFlowNativeModel<Label>) testTrainer(nativeTrainer, trainData, testData);
        List<Prediction<Label>> nativePreds = nativeModel.predict(testData);

        // Check it converts into a checkpoint model and still works
        Path tmpPath = Files.createTempDirectory("tf-native-to-ckpt");
        TensorFlowCheckpointModel<Label> ckptNativeModel = nativeModel.convertToCheckpointModel(tmpPath.toAbsolutePath().toString(),"tribuo-test");
        List<Prediction<Label>> ckptNativePreds = ckptNativeModel.predict(testData);
        checkPredictionEquality(nativePreds,ckptNativePreds);

        // Clean up checkpoint and native model
        ckptNativeModel.close();
        nativeModel.close();
        Files.walk(tmpPath)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);

        // Test checkpoint trainer
        Path checkpointPath = Files.createTempDirectory("tf-classification-test-ckpt");
        TensorFlowTrainer<Label> checkpointTrainer = new TensorFlowTrainer<>(graphDefTuple.graphDef,
                graphDefTuple.outputName,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                denseConverter,
                outputConverter,
                16,
                5,
                16,
                -1,
                checkpointPath);

        // Check the invocation counter works
        TensorFlowCheckpointModel<Label> tmpModel = (TensorFlowCheckpointModel<Label>) testTrainer(checkpointTrainer, trainData, testData);
        tmpModel.close();

        @SuppressWarnings("unchecked")
        TensorFlowCheckpointModel<Label> ckptModel = (TensorFlowCheckpointModel<Label>) testTrainer(checkpointTrainer, trainData, testData);
        List<Prediction<Label>> oldCkptPreds = ckptModel.predict(testData);
        Helpers.testModelProtoSerialization(ckptModel, Label.class, testData);

        // Check we can move the checkpoint and the model still works
        Assertions.assertTrue(ckptModel.isInitialized());
        Path oldPath = Paths.get(ckptModel.getCheckpointDirectory());
        Path newCheckpointPath = oldPath.getParent().resolve("tf-new-path");
        Assertions.assertTrue(oldPath.toFile().renameTo(newCheckpointPath.toFile()));
        ckptModel.setCheckpointDirectory(newCheckpointPath.toString());
        ckptModel.initialize();
        List<Prediction<Label>> ckptPreds = ckptModel.predict(testData);
        checkPredictionEquality(ckptPreds,oldCkptPreds);

        // Check it converts into a native model and still works
        TensorFlowNativeModel<Label> nativeCkptModel = ckptModel.convertToNativeModel();
        List<Prediction<Label>> nativeCkptPreds = nativeCkptModel.predict(testData);
        checkPredictionEquality(ckptPreds,nativeCkptPreds);
        Helpers.testModelProtoSerialization(nativeCkptModel, Label.class, testData);

        // Clean up checkpoint and native model
        ckptModel.close();
        nativeCkptModel.close();

        // Validate that the checkpoint has stored things
        try (Stream<Path> f = Files.list(newCheckpointPath)) {
            List<Path> files = f.collect(Collectors.toList());
            Assertions.assertNotEquals(0, files.size());
        }

        // Cleanup checkpoints
        try (Stream<Path> f = Files.walk(checkpointPath)) {
            f.sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }
        Assertions.assertFalse(Files.exists(checkpointPath));
    }

    private static TensorFlowModel<Label> testTrainer(TensorFlowTrainer<Label> trainer, Dataset<Label> trainData, Dataset<Label> testData) throws IOException {
        // Train the model
        TensorFlowModel<Label> model = trainer.train(trainData);

        // Run smoke test evaluation
        LabelEvaluation eval = new LabelEvaluator().evaluate(model, testData);
        Assertions.assertTrue(eval.averageAUCROC(false) > 0.0);

        // Check Tribuo serialization
        Helpers.testModelSerialization(model, Label.class);

        // Check saved model bundle export
        Path outputPath = Files.createTempDirectory("tf-classification-test");
        model.exportModel(outputPath.toString());
        List<Path> files = Files.list(outputPath).collect(Collectors.toList());
        Assertions.assertNotEquals(0, files.size());

        // Cleanup saved model bundle
        Files.walk(outputPath)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);

        Assertions.assertFalse(Files.exists(outputPath));

        return model;
    }

    private void checkPredictionEquality(List<Prediction<Label>> first, List<Prediction<Label>> second) {
        Assertions.assertEquals(first.size(),second.size());
        for (int i = 0; i < first.size(); i++) {
            Prediction<Label> predA = first.get(i);
            Prediction<Label> predB = second.get(i);
            Assertions.assertTrue(predA.getOutput().fullEquals(predB.getOutput()));
            Assertions.assertTrue(predA.distributionEquals(predB));
        }
    }

    /**
     * Generates image data.
     * <p>
     * The data generating process is as follows:
     * - Compute the number of possible features which could be set. Features are set in a block based on the y
     * co-ordinate which indicates the class label.
     * - Sample a class label y, in the range 0 -> numClasses
     * - For 50% of the number of valid features:
     * -- Randomly sample a feature's y co-ordinate in the range y*pixRange -> (y+1)*pixRange
     * -- Randomly sample the feature's x co-ordinate in the range 0 -> imageSize
     * -- Randomly sample the feature's value in the range (pixelDepth/2,pixelDepth)
     * -- Check if we've added this feature already, if not add it.
     * @param numExamples Number of examples to generate for train and test.
     * @param imageSize The image size in pixels, must be a multiple of the number of classes.
     * @param pixelDepth The number of valid pixel values, must be greater than 1.
     * @param numClasses The number of classes.
     * @param seed The RNG seed.
     * @return Training and test datasets.
     */
    private static Pair<Dataset<Label>,Dataset<Label>> generateImageData(int numExamples, int imageSize, int pixelDepth, int numClasses, int seed) {
        if (imageSize % numClasses != 0) {
            throw new IllegalArgumentException("The data generating process needs imageSize to be a multiple of numClasses.");
        }
        if (pixelDepth < 1) {
            throw new IllegalArgumentException("Pixel depth must be greater than 1");
        }
        SplittableRandom rng = new SplittableRandom(seed);
        LabelFactory factory = new LabelFactory();
        String description = "(numExamples="+numExamples+",imageSize="+imageSize+",pixelDepth="+pixelDepth+",numClasses="+numClasses+",seed="+seed+")";

        int maxFeature = imageSize*imageSize;
        int width = (""+maxFeature).length();
        String formatString = "%0"+width+"d";
        Map<Integer,String> featureNameMap = new HashMap<>(maxFeature);
        for (int i = 0; i < maxFeature; i++) {
            featureNameMap.put(i,String.format(formatString,i));
        }

        int halfDepth = pixelDepth/2;
        int pixRange = imageSize / numClasses;
        int numValidFeatures = pixRange * imageSize;
        List<Example<Label>> trainList = new ArrayList<>();
        Set<String> names = new HashSet<>();
        List<Feature> featuresCache = new ArrayList<>();
        for (int i = 0; i < numExamples; i++) {
            names.clear();
            featuresCache.clear();
            int curLabelIdx = rng.nextInt(numClasses);
            Label curLabel = new Label(""+curLabelIdx);
            for (int j = 0; j < numValidFeatures / 2; j++) {
                int yValue = rng.nextInt(pixRange) + (curLabelIdx * pixRange);
                int xValue = rng.nextInt(imageSize);
                int value = rng.nextInt(halfDepth) + halfDepth;
                // feature name = x*imageSize + y
                int featureIdx = xValue*imageSize + yValue;
                String featureName = featureNameMap.get(featureIdx);
                if (!names.contains(featureName)) {
                    names.add(featureName);
                    featuresCache.add(new Feature(featureName,value));
                }
            }
            trainList.add(new ArrayExample<>(curLabel,featuresCache));
        }

        ListDataSource<Label> trainListSource = new ListDataSource<>(trainList,factory,new SimpleDataSourceProvenance("Training " + description,factory));

        List<Example<Label>> testList = new ArrayList<>();

        for (int i = 0; i < numExamples; i++) {
            names.clear();
            featuresCache.clear();
            int curLabelIdx = rng.nextInt(numClasses);
            Label curLabel = new Label(""+curLabelIdx);
            for (int j = 0; j < numValidFeatures / 2; j++) {
                int yValue = rng.nextInt(pixRange) + (curLabelIdx * pixRange);
                int xValue = rng.nextInt(imageSize);
                int value = rng.nextInt(halfDepth) + halfDepth;
                // feature name = x*imageSize + y
                int featureIdx = xValue*imageSize + yValue;
                String featureName = featureNameMap.get(featureIdx);
                if (!names.contains(featureName)) {
                    names.add(featureName);
                    featuresCache.add(new Feature(featureName,value));
                }
            }
            testList.add(new ArrayExample<>(curLabel,featuresCache));
        }

        ListDataSource<Label> testListSource = new ListDataSource<>(testList,factory,new SimpleDataSourceProvenance("Testing " + description,factory));

        return new Pair<>(new MutableDataset<>(trainListSource),new MutableDataset<>(testListSource));
    }

    @Test
    public void classificationCNNTest() throws IOException {
        // Create the train and test data
        Pair<Dataset<Label>,Dataset<Label>> data = generateImageData(512,10,128,5,42);
        Dataset<Label> trainData = data.getA();
        Dataset<Label> testData = data.getB();

        // Build the CNN graph
        GraphDefTuple graphDefTuple = CNNExamples.buildLeNetGraph(INPUT_NAME, 10, 255, trainData.getOutputs().size());

        // Configure the trainer
        Map<String, Float> gradientParams = new HashMap<>();
        gradientParams.put("learningRate", 0.01f);
        gradientParams.put("initialAccumulatorValue", 0.1f);
        FeatureConverter imageConverter = new ImageConverter(INPUT_NAME, 10, 10, 1);
        OutputConverter<Label> outputConverter = new LabelConverter();
        TensorFlowTrainer<Label> trainer = new TensorFlowTrainer<>(graphDefTuple.graphDef,
                graphDefTuple.outputName,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                imageConverter,
                outputConverter,
                16,
                5,
                16,
                -1);

        // Train the model
        TensorFlowModel<Label> model = trainer.train(trainData);

        // Make some predictions
        List<Prediction<Label>> predictions = model.predict(testData);

        // Run smoke test evaluation
        LabelEvaluation eval = new LabelEvaluator().evaluate(model,predictions,testData.getProvenance());
        Assertions.assertTrue(eval.accuracy() > 0.0);

        // Check Tribuo serialization
        Helpers.testModelSerialization(model,Label.class);

        // Check saved model bundle export
        Path outputPath = Files.createTempDirectory("tf-classification-cnn-test");
        model.exportModel(outputPath.toString());
        try (Stream<Path> f = Files.list(outputPath)) {
            List<Path> files = f.collect(Collectors.toList());
            Assertions.assertNotEquals(0, files.size());
        }

        // Create external model from bundle
        Map<Label,Integer> outputMapping = new HashMap<>();
        for (Pair<Integer,Label> p : model.getOutputIDInfo()) {
            outputMapping.put(p.getB(),p.getA());
        }
        Map<String,Integer> featureMapping = new HashMap<>();
        ImmutableFeatureMap featureIDMap = model.getFeatureIDMap();
        for (VariableInfo info : featureIDMap) {
            featureMapping.put(info.getName(),featureIDMap.getID(info.getName()));
        }
        TensorFlowSavedModelExternalModel<Label> externalModel = TensorFlowSavedModelExternalModel.createTensorflowModel(
                trainData.getOutputFactory(),featureMapping,outputMapping,model.getOutputName(),imageConverter, outputConverter,outputPath.toString());

        // Check predictions are equal
        List<Prediction<Label>> externalPredictions = externalModel.predict(testData);
        checkPredictionEquality(predictions,externalPredictions);
        Helpers.testModelProtoSerialization(externalModel, Label.class, testData);

        // Cleanup saved model bundle
        externalModel.close();
        Files.walk(outputPath)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);

        Assertions.assertFalse(Files.exists(outputPath));

        // Cleanup created model
        model.close();
    }

    private static String ndArrToString(FloatNdArray ndarray) {
        StringBuffer sb = new StringBuffer();
        ndarray.scalars().forEachIndexed((idx,array) -> sb.append(Arrays.toString(idx)).append(" = ").append(array.getFloat()).append("\n"));
        return sb.toString();
    }

    private static String ndArrToString(IntNdArray ndarray) {
        StringBuffer sb = new StringBuffer();
        ndarray.scalars().forEachIndexed((idx,array) -> sb.append(Arrays.toString(idx)).append(" = ").append(array.getInt()).append("\n"));
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        LabelFactory labelFactory = new LabelFactory();
        String base = "./tutorials/";

        System.out.println("Loading data");
        IDXDataSource<Label> trainMNIST = new IDXDataSource<>(Paths.get(base,"train-images-idx3-ubyte.gz"),Paths.get(base,"train-labels-idx1-ubyte.gz"),labelFactory);
        IDXDataSource<Label> testMNIST = new IDXDataSource<>(Paths.get(base,"t10k-images-idx3-ubyte.gz"),Paths.get(base,"t10k-labels-idx1-ubyte.gz"),labelFactory);

        Dataset<Label> train = new MutableDataset<>(trainMNIST);
        Dataset<Label> test = new MutableDataset<>(testMNIST);

        System.out.println("Building graph");
        GraphDefTuple graphDefTuple = CNNExamples.buildLeNetGraph(INPUT_NAME,IMAGE_SIZE,PIXEL_DEPTH,NUM_LABELS);

        System.out.println("Writing graph to " + args[0] + " with output  name '" + graphDefTuple.outputName + "'");
        GraphDef graphDef = graphDefTuple.graphDef;
        byte[] bytes = graphDef.toByteArray();
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(args[0]))) {
            bos.write(bytes);
        }

        Map<String, Float> gradientParams = new HashMap<>();
        gradientParams.put("learningRate", 0.01f);
        gradientParams.put("initialAccumulatorValue", 0.1f);

        FeatureConverter imageConverter = new ImageConverter(INPUT_NAME, 28, 28, 1);
        OutputConverter<Label> outputConverter = new LabelConverter();

        TensorFlowTrainer<Label> trainer = new TensorFlowTrainer<>(graphDefTuple.graphDef,
                graphDefTuple.outputName,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                imageConverter,
                outputConverter,
                16,
                2,
                16,
                1000);

        System.out.println("Training model");
        final long trainStart = System.currentTimeMillis();
        Model<Label> model = trainer.train(train);
        final long trainStop = System.currentTimeMillis();
        System.out.println("Finished training classifier " + Util.formatDuration(trainStart, trainStop));

        System.out.println("Evaluating model");
        LabelEvaluation eval = new LabelEvaluator().evaluate(model,test);

        System.out.println(eval.toString());

        System.out.println(eval.getConfusionMatrix().toString());
    }
}