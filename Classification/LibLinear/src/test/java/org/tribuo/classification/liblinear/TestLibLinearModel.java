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

package org.tribuo.classification.liblinear;

import com.oracle.labs.mlrg.olcut.util.Pair;
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
import org.tribuo.classification.liblinear.LinearClassificationType.LinearType;
import org.tribuo.data.text.TextDataSource;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.data.text.impl.SimpleTextDataSource;
import org.tribuo.data.text.impl.TextFeatureExtractorImpl;
import org.tribuo.dataset.DatasetView;
import org.tribuo.impl.ListExample;
import de.bwaldvogel.liblinear.FeatureNode;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestLibLinearModel {
    private static final Logger logger = Logger.getLogger(TestLibLinearModel.class.getName());

    private static final LibLinearClassificationTrainer t = new LibLinearClassificationTrainer();

    //on Windows, this resolves to some nonsense like this: /C:/workspace/Classification/LibLinear/target/test-classes/test_input.tribuo
    //and the leading slash is a problem and causes this test to fail on windows.
    //it's generally poor practice to convert a resource to a path because the file won't normally exist as a file at runtime
    //it only works at test time because ./target/test-classes/ is a folder that exists and it is on the classpath.
    private final String TEST_INPUT_PATH = this.getClass().getResource("/test_input_binary.tribuo").getPath().replaceFirst("^/(.:/)", "$1");
    private final String TEST_INPUT_PATH_MULTICLASS = this.getClass().getResource("/test_input_multiclass.tribuo").getPath().replaceFirst("^/(.:/)", "$1");

    @Test
    public void testPredictDataset() throws IOException, ClassNotFoundException {
        for (LinearType mtype : LinearType.values()) {
            checkModelType(mtype);
        }
    }

    @Test
    public void testSingleClassTraining() {
        Pair<Dataset<Label>,Dataset<Label>> data = LabelledDataGenerator.denseTrainTest();

        DatasetView<Label> trainingData = DatasetView.createView(data.getA(),(Example<Label> e) -> e.getOutput().getLabel().equals("Foo"), "Foo selector");
        Model<Label> model = t.train(trainingData);
        LabelEvaluation evaluation = (LabelEvaluation) trainingData.getOutputFactory().getEvaluator().evaluate(model,data.getB());
        assertEquals(0.0,evaluation.accuracy(new Label("Bar")));
        assertEquals(0.0,evaluation.accuracy(new Label("Baz")));
        assertEquals(0.0,evaluation.accuracy(new Label("Quux")));
        assertEquals(1.0,evaluation.recall(new Label("Foo")));
    }

    @Test
    public void testMulticlass() throws IOException, ClassNotFoundException {
        String prefix = "L2R_LR_multiclass";
        LibLinearClassificationModel model = loadModel("/models/L2R_LR_multiclass.model");
        Dataset<Label> examples = loadMulticlassTestDataset(model);
        assertNotNull(model, prefix);
        List<Prediction<Label>> predictions = model.predict(examples);
        assertEquals(predictions.size(), examples.size(), prefix);
        for (Prediction<Label> p : predictions) {
            checkPrediction(prefix, model, p);
        }
        // check for ArrayIndexOutOfBounds
        for (Example<Label> example : examples) {
            model.getExcuse(example);
        }
    }

    private void checkModelType(LinearType modelType) throws IOException, ClassNotFoundException {
        String prefix = String.format("model %s", modelType);
        LibLinearClassificationModel model = loadModel(modelType);
        Dataset<Label> examples = loadTestDataset(model);
        assertNotNull(model, prefix);
        List<Prediction<Label>> predictions = model.predict(examples);
        assertEquals(predictions.size(), examples.size(), prefix);
        for (Prediction<Label> p : predictions) {
            checkPrediction(prefix, model, p);
        }
        // check for ArrayIndexOutOfBounds
        for (Example<Label> example : examples) {
            model.getExcuse(example);
        }
    }


    private LibLinearClassificationModel loadModel(LinearType modelType) throws IOException, ClassNotFoundException {
        String modelPath = "/models/" + modelType + ".model";
        return loadModel(modelPath);
    }

    private LibLinearClassificationModel loadModel(String path) throws IOException, ClassNotFoundException {
        File modelFile = new File(this.getClass().getResource(path).getPath());
        assertTrue(modelFile.exists(),String.format("model for %s does not exist", path));
        try (ObjectInputStream oin = new ObjectInputStream(new FileInputStream(modelFile))) {
            Object data = oin.readObject();
            return (LibLinearClassificationModel) data;
        }
    }

    private Dataset<Label> loadTestDataset(LibLinearClassificationModel model) throws IOException {
        return loadDataset(model, Paths.get(TEST_INPUT_PATH));
    }

    private Dataset<Label> loadMulticlassTestDataset(LibLinearClassificationModel model) throws IOException {
        return loadDataset(model, Paths.get(TEST_INPUT_PATH_MULTICLASS));
    }

    private Dataset<Label> loadDataset(LibLinearClassificationModel model, Path path) throws IOException {
        TextFeatureExtractor<Label> extractor = new TextFeatureExtractorImpl<>(new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2));
        TextDataSource<Label> src = new SimpleTextDataSource<>(path, new LabelFactory(), extractor);
        return new ImmutableDataset<>(src, model.getFeatureIDMap(), model.getOutputIDInfo(), false);
    }

    private void checkPrediction(String msgPrefix, LibLinearClassificationModel model, Prediction<Label> prediction) {
        assertNotNull(prediction);
        ImmutableOutputInfo<Label> labelMap = model.getOutputIDInfo();
        Map<String,Label> dist = prediction.getOutputScores();
        for (Label k : labelMap.getDomain()) {
            String msg = String.format("%s --> dist did not contain entry for label %s", msgPrefix, k);
            assertTrue(dist.containsKey(k.getLabel()), msg);
        }
    }

    public Model<Label> testLibLinear(Pair<Dataset<Label>,Dataset<Label>> p) {
        Model<Label> m = t.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        return m;
    }

    @Test
    public void testReproducible() {
        // Note this test will need to change if LibLinearTrainer grows a per Problem RNG.
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> m = t.train(p.getA());
        Map<String, List<Pair<String,Double>>> mFeatures = m.getTopFeatures(-1);

        Model<Label> mTwo = t.train(p.getA());
        Map<String, List<Pair<String,Double>>> mTwoFeatures = mTwo.getTopFeatures(-1);

        assertEquals(mFeatures,mTwoFeatures);
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testLibLinear(p);

        // Test serialization
        Helpers.testModelSerialization(model,Label.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testLibLinear(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testLibLinear(p);
    }

    @Test
    public void duplicateFeatureIDs() {
        ImmutableFeatureMap fmap = new TestMap();

        Example<Label> collision = generateExample(new String[]{"FOO","BAR","BAZ","QUUX"},new double[]{1.0,2.2,3.3,4.4});
        int[] testCollisionIndices = new int[]{1,2,3,4};
        double[] testCollisionValues = new double[]{4.3,2.2,4.4,1.0};
        FeatureNode[] nodes = LibLinearClassificationTrainer.exampleToNodes(collision,fmap,null);
        int[] nodesIndices = getIndices(nodes);
        double[] nodesValues = getValues(nodes);
        logger.log(Level.FINE,"node values " + Arrays.toString(nodes));
        assertArrayEquals(testCollisionIndices,nodesIndices);
        assertArrayEquals(testCollisionValues,nodesValues,1e-10);

        Example<Label> fakecollision = generateExample(new String[]{"BAR","BAZ","QUUX"},new double[]{2.2,3.3,4.4});
        testCollisionIndices = new int[]{1,2,3,4};
        testCollisionValues = new double[]{3.3,2.2,4.4,1.0};
        nodes = LibLinearClassificationTrainer.exampleToNodes(fakecollision,fmap,null);
        nodesIndices = getIndices(nodes);
        nodesValues = getValues(nodes);
        logger.log(Level.FINE,"node values " + Arrays.toString(nodes));
        assertArrayEquals(testCollisionIndices,nodesIndices);
        assertArrayEquals(testCollisionValues,nodesValues,1e-10);
    }

    @Test
    public void testInvalidExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = t.train(p.getA());
            m.predict(LabelledDataGenerator.invalidSparseExample());
        });
    }

    @Test
    public void testEmptyExample() {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Label>, Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
            Model<Label> m = t.train(p.getA());
            m.predict(LabelledDataGenerator.emptyExample());
        });
    }

    private static int[] getIndices(FeatureNode[] nodes) {
        int[] indices = new int[nodes.length];

        for (int i = 0; i < nodes.length; i++) {
            indices[i] = nodes[i].index;
        }

        return indices;
    }

    private static double[] getValues(FeatureNode[] nodes) {
        double[] values = new double[nodes.length];

        for (int i = 0; i < nodes.length; i++) {
            values[i] = nodes[i].value;
        }

        return values;
    }

    private static Example<Label> generateExample(String[] names, double[] values) {
        Example<Label> e = new ListExample<>(new Label("MONKEYS"));
        for (int i = 0; i < names.length; i++) {
            e.add(new Feature(names[i],values[i]));
        }
        return e;
    }

    private static class TestMap extends ImmutableFeatureMap {
        private static final long serialVersionUID = 1L;
        public TestMap() {
            super();
            CategoricalIDInfo foo = (new CategoricalInfo("FOO")).makeIDInfo(0);
            m.put("FOO",foo);
            idMap.put(0,foo);
            CategoricalIDInfo bar = (new CategoricalInfo("BAR")).makeIDInfo(1);
            m.put("BAR",bar);
            idMap.put(1,bar);
            CategoricalIDInfo baz = (new CategoricalInfo("BAZ")).makeIDInfo(0);
            m.put("BAZ",baz);
            idMap.put(0,baz);
            CategoricalIDInfo quux = (new CategoricalInfo("QUUX")).makeIDInfo(2);
            m.put("QUUX",quux);
            idMap.put(2,quux);
            size = idMap.size();
        }
    }

}
