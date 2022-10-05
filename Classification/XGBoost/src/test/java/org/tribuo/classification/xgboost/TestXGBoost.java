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

package org.tribuo.classification.xgboost;

import com.oracle.labs.mlrg.olcut.provenance.MapProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
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
import org.tribuo.common.xgboost.XGBoostTrainer;
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

import java.io.IOException;
import java.io.ObjectInputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class TestXGBoost {

    private static final XGBoostClassificationTrainer t = new XGBoostClassificationTrainer(50);

    private static final XGBoostClassificationTrainer dart = new XGBoostClassificationTrainer(
            XGBoostTrainer.BoosterType.DART,XGBoostTrainer.TreeMethod.AUTO,50,0.3,0,6,1,1,1,1,0,1, XGBoostTrainer.LoggingVerbosity.SILENT,42);

    private static final XGBoostClassificationTrainer linear = new XGBoostClassificationTrainer(
            XGBoostTrainer.BoosterType.LINEAR,XGBoostTrainer.TreeMethod.AUTO,50,0.3,0,6,1,1,1,1,0,1, XGBoostTrainer.LoggingVerbosity.SILENT,42);

    private static final XGBoostClassificationTrainer gbtree = new XGBoostClassificationTrainer(
            XGBoostTrainer.BoosterType.GBTREE,XGBoostTrainer.TreeMethod.HIST,50,0.3,0,6,1,1,1,1,0,1, XGBoostTrainer.LoggingVerbosity.SILENT,42);

    private static final int[] NUM_TREES = new int[]{1,5,10,50};

    private static final Path TEST_INPUT_PATH;
    private static final Path TEST_INPUT_PATH_MULTICLASS;
    static {
        URL input = null;
        try {
            input = TestXGBoost.class.getResource("/test_input_binary.tribuo");
            TEST_INPUT_PATH = Paths.get(input.toURI());
            input = TestXGBoost.class.getResource("/test_input_multiclass.tribuo");
            TEST_INPUT_PATH_MULTICLASS = Paths.get(input.toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Invalid URL to test resource " + input);
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
        URL modelFile = this.getClass().getResource(path);
        try (ObjectInputStream ois = new ObjectInputStream(modelFile.openStream())) {
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
        return loadDataset(model, TEST_INPUT_PATH);
    }

    private Dataset<Label> loadMulticlassTestDataset(XGBoostModel<Label> model) throws IOException {
        return loadDataset(model, TEST_INPUT_PATH_MULTICLASS);
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

    public static XGBoostModel<Label> testXGBoost(XGBoostClassificationTrainer trainer, Pair<Dataset<Label>,Dataset<Label>> p) {
        XGBoostModel<Label> m = trainer.train(p.getA());
        LabelEvaluator e = new LabelEvaluator();
        LabelEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        assertFalse(features.isEmpty());
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
        Model<Label> model = testXGBoost(t,p);
        Helpers.testModelSerialization(model,Label.class);
        Helpers.testModelProtoSerialization(model, Label.class, p.getB());
        testXGBoost(dart,p);
        testXGBoost(linear,p);
        XGBoostModel<Label> m = testXGBoost(gbtree,p);

        // Check params are not overridden by default
        @SuppressWarnings("unchecked")
        MapProvenance<StringProvenance> params = (MapProvenance<StringProvenance>) m.getProvenance().getTrainerProvenance().getConfiguredParameters().get("overrideParameters");
        assertTrue(params.getMap().isEmpty());

        // Check overridden params are the right size
        Map<String,Object> overrideParams = new HashMap<>();
        overrideParams.put("objective","multi:softprob");
        overrideParams.put("eta","0.1");
        overrideParams.put("sampling_method","gradient_based");
        XGBoostClassificationTrainer overrideTrainer = new XGBoostClassificationTrainer(5, overrideParams);
        XGBoostModel<Label> overrideM = testXGBoost(overrideTrainer,p);

        @SuppressWarnings("unchecked")
        MapProvenance<StringProvenance> overrideProvMapParams = (MapProvenance<StringProvenance>) overrideM.getProvenance().getTrainerProvenance().getConfiguredParameters().get("overrideParameters");
        Map<String, StringProvenance> overrideMap = overrideProvMapParams.getMap();
        assertEquals(overrideParams.size(), overrideMap.size());
        assertEquals(overrideParams.get("objective"), overrideMap.get("objective").getValue());
        assertEquals(overrideParams.get("eta"), overrideMap.get("eta").getValue());
        assertEquals(overrideParams.get("sampling_method"), overrideMap.get("sampling_method").getValue());
    }

    @Test
    public void testSparseData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.sparseTrainTest();
        testXGBoost(t,p);
    }

    @Test
    public void testSparseBinaryData() {
        Pair<Dataset<Label>,Dataset<Label>> p = LabelledDataGenerator.binarySparseTrainTest();
        testXGBoost(t,p);
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
