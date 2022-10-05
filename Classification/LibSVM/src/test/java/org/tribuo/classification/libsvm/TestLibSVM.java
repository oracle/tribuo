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

package org.tribuo.classification.libsvm;

import ai.onnxruntime.OrtException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import libsvm.svm_model;
import libsvm.svm_node;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.CategoricalIDInfo;
import org.tribuo.CategoricalInfo;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.classification.libsvm.SVMClassificationType.SVMMode;
import org.tribuo.common.libsvm.KernelType;
import org.tribuo.common.libsvm.LibSVMModel;
import org.tribuo.common.libsvm.LibSVMTrainer;
import org.tribuo.common.libsvm.SVMParameters;
import org.tribuo.data.text.TextDataSource;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.data.text.impl.SimpleTextDataSource;
import org.tribuo.data.text.impl.TextFeatureExtractorImpl;
import org.tribuo.dataset.DatasetView;
import org.tribuo.impl.ListExample;
import org.tribuo.interop.onnx.OnnxTestUtils;
import org.tribuo.test.Helpers;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class TestLibSVM {
    private static final Logger logger = Logger.getLogger(TestLibSVM.class.getName());

    private static final LibSVMClassificationTrainer C_RBF = new LibSVMClassificationTrainer(new SVMParameters<>(new SVMClassificationType(SVMMode.C_SVC), KernelType.RBF));
    private static final LibSVMClassificationTrainer NU_RBF = new LibSVMClassificationTrainer(new SVMParameters<>(new SVMClassificationType(SVMMode.NU_SVC), KernelType.RBF));
    private static final LibSVMClassificationTrainer C_LINEAR = new LibSVMClassificationTrainer(new SVMParameters<>(new SVMClassificationType(SVMMode.C_SVC), KernelType.LINEAR));
    private static final LibSVMClassificationTrainer NU_LINEAR = new LibSVMClassificationTrainer(new SVMParameters<>(new SVMClassificationType(SVMMode.NU_SVC), KernelType.LINEAR));

    private static final Path TEST_INPUT_PATH;
    private static final Path TEST_INPUT_PATH_MULTICLASS;
    static {
        URL input = null;
        try {
            input = TestLibSVM.class.getResource("/test_input_binary.tribuo");
            TEST_INPUT_PATH = Paths.get(input.toURI());
            input = TestLibSVM.class.getResource("/test_input_multiclass.tribuo");
            TEST_INPUT_PATH_MULTICLASS = Paths.get(input.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Invalid URL to test resource " + input);
        }
    }

    @Test
    public void testSingleClassTraining() {
        Pair<Dataset<Label>, Dataset<Label>> data = LabelledDataGenerator.denseTrainTest();

        DatasetView<Label> trainingData = DatasetView.createView(data.getA(), (Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");
        Model<Label> model = C_RBF.train(trainingData);
        LabelEvaluation evaluation = (LabelEvaluation) trainingData.getOutputFactory().getEvaluator().evaluate(model, data.getB());
        assertEquals(0.0, evaluation.accuracy(new Label("Bar")));
        assertEquals(0.0, evaluation.accuracy(new Label("Baz")));
        assertEquals(0.0, evaluation.accuracy(new Label("Quux")));
        assertEquals(1.0, evaluation.recall(new Label("Foo")));
    }

    @Test
    public void testPredictDataset() throws IOException, ClassNotFoundException {
        for (KernelType kType : KernelType.values()) {
            checkModelType(SVMClassificationType.SVMMode.C_SVC, kType);
            checkModelType(SVMClassificationType.SVMMode.NU_SVC, kType);
        }
    }

    private void checkModelType(SVMMode modelType, KernelType kernelType) throws IOException, ClassNotFoundException {
        String prefix = String.format("model %s-%s", modelType, kernelType);
        LibSVMModel<Label> model = loadModel(modelType, kernelType, false);
        Dataset<Label> examples = loadTestDataset(model);
        assertNotNull(model, prefix);
        List<Prediction<Label>> predictions = model.predict(examples);
        assertEquals(predictions.size(), examples.size(), prefix);
        for (Prediction<Label> p : predictions) {
            checkPrediction(prefix, model, p);
        }
        // check for ArrayIndexOutOfBounds
        for (Example<Label> e : examples) {
            model.getExcuse(e);
        }
        model = loadModel(modelType, kernelType, true);
        examples = loadMulticlassTestDataset(model);
        assertNotNull(model, prefix);
        predictions = model.predict(examples);
        assertEquals(predictions.size(), examples.size(), prefix);
        for (Prediction<Label> p : predictions) {
            checkPrediction(prefix, model, p);
        }
        // check for ArrayIndexOutOfBounds
        for (Example<Label> e : examples) {
            model.getExcuse(e);
        }
        //System.out.println("*** PASSED: " + prefix);
    }


    private LibSVMModel<Label> loadModel(SVMClassificationType.SVMMode modelType, KernelType kernelType, boolean multiclass) throws IOException, ClassNotFoundException {
        String modelPath = "/models/" + modelType + "_" + kernelType;
        if (multiclass) {
            modelPath += "_multiclass";
        }
        modelPath += ".model";
        return loadModel(modelPath);
    }

    private LibSVMModel<Label> loadModel(String path) throws IOException, ClassNotFoundException {
        URL modelFile = this.getClass().getResource(path);
        try (ObjectInputStream ois = new ObjectInputStream(modelFile.openStream())) {
            Object data = ois.readObject();
            return (LibSVMClassificationModel) data;
        } catch (NullPointerException e) {
            fail(String.format("model for %s does not exist", path));
            throw e;
        }
    }

    private Dataset<Label> loadTestDataset(LibSVMModel<Label> model) throws IOException {
        return loadDataset(model, TEST_INPUT_PATH);
    }

    private Dataset<Label> loadMulticlassTestDataset(LibSVMModel<Label> model) throws IOException {
        return loadDataset(model, TEST_INPUT_PATH_MULTICLASS);
    }

    private Dataset<Label> loadDataset(LibSVMModel<Label> model, Path path) throws IOException {
        TextFeatureExtractor<Label> extractor = new TextFeatureExtractorImpl<>(new BasicPipeline(new BreakIteratorTokenizer(Locale.US), 2));
        TextDataSource<Label> src = new SimpleTextDataSource<>(path, new LabelFactory(), extractor);
        return new ImmutableDataset<>(src, model.getFeatureIDMap(), model.getOutputIDInfo(), false);
    }

    private void checkPrediction(String msgPrefix, LibSVMModel<Label> model, Prediction<Label> prediction) {
        assertNotNull(prediction);
        ImmutableOutputInfo<Label> labelMap = model.getOutputIDInfo();
        Map<String, Label> dist = prediction.getOutputScores();
        for (Label k : labelMap.getDomain()) {
            String msg = String.format("%s --> dist did not contain entry for label %s", msgPrefix, k);
            assertTrue(dist.containsKey(k.getLabel()), msg);
        }
    }

    public Model<Label> testLibSVM(Pair<Dataset<Label>, Dataset<Label>> p) {
        Model<Label> m = C_RBF.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m, p.getB());
        Map<String, List<Pair<String, Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertTrue(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertTrue(features.isEmpty());
        return m;
    }

    @Test
    public void testReproducibility() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        long seed = 42L;
        SVMParameters<Label> params = new SVMParameters<>(new SVMClassificationType(SVMMode.NU_SVC), KernelType.RBF);
        params.setProbability();
        LibSVMTrainer<Label> first = new LibSVMClassificationTrainer(params, seed);
        LibSVMModel<Label> firstModel = first.train(p.getA());

        LibSVMTrainer<Label> second = new LibSVMClassificationTrainer(params, seed);
        LibSVMModel<Label> secondModel = second.train(p.getA());

        LibSVMModel<Label> thirdModel = first.train(p.getA());
        LibSVMModel<Label> fourthModel = second.train(p.getA());

        svm_model m = firstModel.getInnerModels().get(0);
        svm_model mTwo = secondModel.getInnerModels().get(0);
        svm_model mThre = thirdModel.getInnerModels().get(0);
        svm_model mFour = fourthModel.getInnerModels().get(0);

        // One and two use the same RNG seed and should be identical
        assertArrayEquals(m.sv_coef, mTwo.sv_coef);
        assertArrayEquals(m.probA, mTwo.probA);
        assertArrayEquals(m.probB, mTwo.probB);

        // The RNG state of three has diverged and should produce a different model.
        assertFalse(Arrays.equals(mTwo.probA, mFour.probA));
        assertFalse(Arrays.equals(mTwo.probB, mFour.probB));

        // The RNG state for three and four are the same so the two models should be the same.
        assertArrayEquals(mFour.sv_coef, mThre.sv_coef);
        assertArrayEquals(mFour.probA, mThre.probA);
        assertArrayEquals(mFour.probB, mThre.probB);
    }

    @Test
    public void testOnnxSerialization() throws IOException, OrtException {
        Pair<Dataset<Label>, Dataset<Label>> binary = LabelledDataGenerator.binarySparseTrainTest();

        Map<Label,Integer> mapping = new HashMap<>();
        mapping.put(new Label("Foo"),0);
        mapping.put(new Label("Bar"),1);
        ImmutableOutputInfo<Label> newInfo = new LabelFactory().constructInfoForExternalModel(mapping);

        ImmutableDataset<Label> newTrain = ImmutableDataset.copyDataset(binary.getA(), binary.getA().getFeatureIDMap(), newInfo);
        ImmutableDataset<Label> newTest = ImmutableDataset.copyDataset(binary.getB(), binary.getA().getFeatureIDMap(), newInfo);

        testSerialization(new Pair<>(newTrain,newTest), C_LINEAR);

        testSerialization(binary, C_LINEAR);
        testSerialization(binary, C_RBF);
        testSerialization(binary, NU_LINEAR);
        testSerialization(binary, NU_RBF);

        SVMParameters<Label> params = new SVMParameters<>(new SVMClassificationType(SVMMode.NU_SVC), KernelType.RBF);
        params.setProbability();
        LibSVMClassificationTrainer probTrainer = new LibSVMClassificationTrainer(params);

        testSerialization(binary,probTrainer);
    }

    @Test
    public void testOnnxMulticlassSerialization() throws IOException, OrtException {
        Pair<Dataset<Label>,Dataset<Label>> multiclass = LabelledDataGenerator.denseTrainTest();

        testSerialization(multiclass,C_LINEAR);
        testSerialization(multiclass,C_RBF);
        testSerialization(multiclass,NU_LINEAR);
        testSerialization(multiclass,NU_RBF);

        SVMParameters<Label> params = new SVMParameters<>(new SVMClassificationType(SVMMode.NU_SVC), KernelType.RBF);
        params.setProbability();
        LibSVMClassificationTrainer probTrainer = new LibSVMClassificationTrainer(params);

        testSerialization(multiclass,probTrainer);
    }

    private static void testSerialization(Pair<Dataset<Label>,Dataset<Label>> datasetPair, LibSVMClassificationTrainer trainer) throws IOException, OrtException {
        LibSVMClassificationModel model = (LibSVMClassificationModel) trainer.train(datasetPair.getA());

        // Test protobuf serialization
        Helpers.testModelProtoSerialization(model, Label.class, datasetPair.getB());

        // Write out model in ONNX
        Path onnxFile = Files.createTempFile("tribuo-libsvm-test", ".onnx");
        model.saveONNXModel("org.tribuo.classification.libsvm.test", 1, onnxFile);

        OnnxTestUtils.onnxLabelComparison(model,onnxFile,datasetPair.getB(),1e-3);

        onnxFile.toFile().delete();
    }


    @Test
    public void testDenseData() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testLibSVM(p);
        Helpers.testModelSerialization(model, Label.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testLibSVM(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testLibSVM(p);
    }

    @Test
    public void duplicateFeatureIDs() {
        ImmutableFeatureMap fmap = new TestMap();

        Example<Label> collision = generateExample(new String[]{"FOO", "BAR", "BAZ", "QUUX"}, new double[]{1.0, 2.2, 3.3, 4.4});
        int[] testCollisionIndices = new int[]{0, 1, 2};
        double[] testCollisionValues = new double[]{4.3, 2.2, 4.4};
        svm_node[] nodes = LibSVMTrainer.exampleToNodes(collision, fmap, null);
        int[] nodesIndices = getIndices(nodes);
        double[] nodesValues = getValues(nodes);
        assertArrayEquals(testCollisionIndices, nodesIndices);
        assertArrayEquals(testCollisionValues, nodesValues, 1e-10);

        Example<Label> fakecollision = generateExample(new String[]{"BAR", "BAZ", "QUUX"}, new double[]{2.2, 3.3, 4.4});
        testCollisionIndices = new int[]{0, 1, 2};
        testCollisionValues = new double[]{3.3, 2.2, 4.4};
        nodes = LibSVMTrainer.exampleToNodes(fakecollision, fmap, null);
        nodesIndices = getIndices(nodes);
        nodesValues = getValues(nodes);
        assertArrayEquals(testCollisionIndices, nodesIndices);
        assertArrayEquals(testCollisionValues, nodesValues, 1e-10);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = C_RBF.train(p.getA());
            m.predict(LabelledDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = C_RBF.train(p.getA());
            m.predict(LabelledDataGenerator.emptyExample());
        });
    }

    private static int[] getIndices(svm_node[] nodes) {
        int[] indices = new int[nodes.length];

        for (int i = 0; i < indices.length; i++) {
            indices[i] = nodes[i].index;
        }

        return indices;
    }

    private static double[] getValues(svm_node[] nodes) {
        double[] values = new double[nodes.length];

        for (int i = 0; i < values.length; i++) {
            values[i] = nodes[i].value;
        }

        return values;
    }

    private static Example<Label> generateExample(String[] names, double[] values) {
        Example<Label> e = new ListExample<>(new Label("MONKEYS"));
        for (int i = 0; i < names.length; i++) {
            e.add(new Feature(names[i], values[i]));
        }
        return e;
    }

    private static class TestMap extends ImmutableFeatureMap {
        private static final long serialVersionUID = 1L;

        public TestMap() {
            super();
            CategoricalIDInfo foo = (new CategoricalInfo("FOO")).makeIDInfo(0);
            m.put("FOO", foo);
            idMap.put(0, foo);
            CategoricalIDInfo bar = (new CategoricalInfo("BAR")).makeIDInfo(1);
            m.put("BAR", bar);
            idMap.put(1, bar);
            CategoricalIDInfo baz = (new CategoricalInfo("BAZ")).makeIDInfo(0);
            m.put("BAZ", baz);
            idMap.put(0, baz);
            CategoricalIDInfo quux = (new CategoricalInfo("QUUX")).makeIDInfo(2);
            m.put("QUUX", quux);
            idMap.put(2, quux);
            size = idMap.size();
        }
    }
}
