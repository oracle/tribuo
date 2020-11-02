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

package org.tribuo.classification.xgboost;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableDataset;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.classification.example.LabelledDataGenerator;
import org.tribuo.common.xgboost.XGBoostFeatureImportance;
import org.tribuo.common.xgboost.XGBoostModel;
import org.tribuo.data.text.TextDataSource;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.data.text.impl.SimpleTextDataSource;
import org.tribuo.data.text.impl.TextFeatureExtractorImpl;
import org.tribuo.dataset.DatasetView;
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
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class TestXGBoost {

    private static final XGBoostClassificationTrainer t = new XGBoostClassificationTrainer(50);

    private static final int[] NUM_TREES = new int[]{1,5,10,50};

    //on Windows, this resolves to some nonsense like this: /C:/workspace/Classification/XGBoost/target/test-classes/test_input.tribuo
    //and the leading slash is a problem and causes this test to fail on windows.
    //it's generally poor practice to convert a resource to a path because the file won't normally exist as a file at runtime
    //it only works at test time because ./target/test-classes/ is a folder that exists and it is on the classpath.
    private final String TEST_INPUT_PATH = this.getClass().getResource("/test_input_binary.tribuo").getPath().replaceFirst("^/(.:/)", "$1");
    private final String TEST_INPUT_PATH_MULTICLASS = this.getClass().getResource("/test_input_multiclass.tribuo").getPath().replaceFirst("^/(.:/)", "$1");

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
    public void testPredictDataset() throws IOException, ClassNotFoundException {
        for (int numTrees : NUM_TREES) {
            checkModelType(numTrees);
        }
    }

    private void checkModelType(int numTrees) throws IOException, ClassNotFoundException {
        String prefix = String.format("model %s", numTrees);
        XGBoostModel<Label> model = loadModel(numTrees,false);
        Dataset<Label> examples = loadTestDataset(model);
        assertNotNull(model, prefix);
        List<Prediction<Label>> predictions = model.predict(examples);
        assertEquals(predictions.size(), examples.size(), prefix);
        for (Prediction<Label> p : predictions) {
            checkPrediction(prefix, model, p);
        }
        // check for ArrayIndexOutOfBounds
        for (Example<Label> features : examples) {
            model.getExcuse(features);
        }
        model = loadModel(numTrees,true);
        examples = loadMulticlassTestDataset(model);
        assertNotNull(model, prefix);
        predictions = model.predict(examples);
        assertEquals(predictions.size(), examples.size(), prefix);
        for (Prediction<Label> p : predictions) {
            checkPrediction(prefix, model, p);
        }
        // check for ArrayIndexOutOfBounds
        for (Example<Label> example : examples) {
            model.getExcuse(example);
        }
        //System.out.println("*** PASSED: " + prefix);
    }


    private XGBoostModel<Label> loadModel(int numTrees, boolean multiclass) throws IOException, ClassNotFoundException {
        String modelPath = "/models/" + numTrees;
        if (multiclass) {
            modelPath += "_multiclass";
        }
        modelPath += ".model";
        return loadModel(modelPath);
    }

    private XGBoostModel<Label> loadModel(String path) throws IOException, ClassNotFoundException {
        File modelFile = new File(this.getClass().getResource(path).getPath());
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile))) {
            @SuppressWarnings("unchecked") // checked by validate call.
            XGBoostModel<Label> data = (XGBoostModel<Label>) ois.readObject();
            if (!data.validate(Label.class)) {
                fail(String.format("model for %s is not a classification model.",path));
            }
            return data;
        } catch (NullPointerException e) {
            fail(String.format("model for %s does not exist", path));
            throw e;
        }
    }

    private Dataset<Label> loadTestDataset(XGBoostModel<Label> model) throws IOException {
        return loadDataset(model, Paths.get(TEST_INPUT_PATH));
    }

    private Dataset<Label> loadMulticlassTestDataset(XGBoostModel<Label> model) throws IOException {
        return loadDataset(model, Paths.get(TEST_INPUT_PATH_MULTICLASS));
    }

    private Dataset<Label> loadDataset(XGBoostModel<Label> model, Path path) throws IOException {
        TextFeatureExtractor<Label> extractor = new TextFeatureExtractorImpl<>(new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2));
        TextDataSource<Label> src = new SimpleTextDataSource<>(path, new LabelFactory(), extractor);
        return new ImmutableDataset<>(src, model.getFeatureIDMap(), model.getOutputIDInfo(),false);
    }

    private void checkPrediction(String msgPrefix, XGBoostModel<Label> model, Prediction<Label> prediction) {
        assertNotNull(prediction);
        ImmutableOutputInfo<Label> labelMap = model.getOutputIDInfo();
        Map<String,Label> dist = prediction.getOutputScores();
        for (Label k : labelMap.getDomain()) {
            String msg = String.format("%s --> dist did not contain entry for label %s", msgPrefix, k);
            assertTrue(dist.containsKey(k.getLabel()), msg);
        }
    }

    public Model<Label> testXGBoost(Pair<Dataset<Label>,Dataset<Label>> p) {
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
    public void testFeatureImportanceSmokeTest() {
        // we're just testing that not actually throws an exception
        XGBoostModel<Label> m = (XGBoostModel<Label>)t.train(LabelledDataGenerator.denseTrainTest().getA());

        XGBoostFeatureImportance i = m.getFeatureImportance().get(0);
        i.getImportances();
        i.getCover();
        i.getGain();
        i.getWeight();
        i.getTotalCover();
        i.getTotalGain();

        i.getImportances(5);
        i.getCover(5);
        i.getGain(5);
        i.getWeight(5);
        i.getTotalCover(5);
        i.getTotalGain(5);
    }

    @Test
    public void testDenseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.denseTrainTest();
        Model<Label> model = testXGBoost(p);
        Helpers.testModelSerialization(model,Label.class);
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testXGBoost(p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testXGBoost(p);
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
}
