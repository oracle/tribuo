/*
 * Copyright (c) 2015, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.rtree;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.SparseModel;
import org.tribuo.Trainer;
import org.tribuo.common.tree.TreeModel;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.RegressionDataGenerator;
import org.tribuo.regression.rtree.impurity.MeanSquaredError;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestCARTRegressionTrainer {

    private static final RegressionEvaluator e = new RegressionEvaluator();
    private static final CARTRegressionTrainer t = new CARTRegressionTrainer();
    private static final CARTRegressionTrainer randomt = new CARTRegressionTrainer(Integer.MAX_VALUE, 5, 0.0f, 0.75f,
        true,
            new MeanSquaredError(), Trainer.DEFAULT_SEED);

    public static Model<Regressor> testIndependentRegressionTree(Pair<Dataset<Regressor>,Dataset<Regressor>> p,
                                              CARTRegressionTrainer trainer) {
        Model<Regressor> m = trainer.train(p.getA());
        RegressionEvaluation evaluation = e.evaluate(m,p.getB());
        Map<String, List<Pair<String,Double>>> features = m.getTopFeatures(3);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        features = m.getTopFeatures(-1);
        Assertions.assertNotNull(features);
        Assertions.assertFalse(features.isEmpty());
        Helpers.testModelProtoSerialization(m, Regressor.class, p.getB());
        return m;
    }

    public static Model<Regressor> runDenseData(CARTRegressionTrainer trainer) {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
        return testIndependentRegressionTree(p, trainer);
    }

    @Test
    public void testDenseData() {
        Model<Regressor> model = runDenseData(t);
        Helpers.testModelProtoSerialization(model,Regressor.class);
    }

    @Test
    public void testRandomDenseData() {
        runDenseData(randomt);
    }

    public void runSparseData(CARTRegressionTrainer trainer) {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.sparseTrainTest();
        testIndependentRegressionTree(p, trainer);
    }

    @Test
    public void testSparseData() {
        runSparseData(t);
    }

    @Test
    public void testRandomSparseData() {
        runSparseData(randomt);
    }

    public void runInvalidExample(CARTRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.invalidMultiDimSparseExample());
        });
    }

    @Test
    public void testInvalidExample() {
        runInvalidExample(t);
    }

    @Test
    public void testRandomInvalidExample() {
        runInvalidExample(randomt);
    }

    public void runEmptyExample(CARTRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.denseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.emptyMultiDimExample());
        });
    }

    @Test
    public void testEmptyExample() {
        runEmptyExample(t);
    }

    @Test
    public void testRandomEmptyExample() {
        runEmptyExample(randomt);
    }

    public void runMultiDenseData(CARTRegressionTrainer trainer) {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
        testIndependentRegressionTree(p, trainer);
    }

    @Test
    public void testMultiDenseData() {
        runMultiDenseData(t);
    }

    @Test
    public void testMultiRandomDenseData() {
        runMultiDenseData(randomt);
    }

    public void runMultiSparseData(CARTRegressionTrainer trainer) {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.multiDimSparseTrainTest();
        testIndependentRegressionTree(p, trainer);
    }

    @Test
    public void testMultiSparseData() {
        runMultiSparseData(t);
    }

    @Test
    public void testRandomMultiSparseData() {
        runMultiSparseData(randomt);
    }

    public void runMultiInvalidExample(CARTRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.invalidMultiDimSparseExample());
        });
    }

    @Test
    public void testMultiInvalidExample() {
        runMultiInvalidExample(t);
    }

    @Test
    public void testRandomMultiInvalidExample() {
        runMultiInvalidExample(randomt);
    }

    public void runMultiEmptyExample(CARTRegressionTrainer trainer) {
        assertThrows(IllegalArgumentException.class, () -> {
            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.multiDimDenseTrainTest();
            Model<Regressor> m = trainer.train(p.getA());
            m.predict(RegressionDataGenerator.emptyMultiDimExample());
        });
    }

    @Test
    public void testMultiEmptyExample() {
        runMultiEmptyExample(t);
    }

    @Test
    public void testRandomMultiEmptyExample() {
        runMultiEmptyExample(randomt);
    }

    @Test
    public void testThreeDenseData() {
        Pair<Dataset<Regressor>,Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        TreeModel<Regressor> llModel = t.train(p.getA());
        RegressionEvaluation llEval = e.evaluate(llModel,p.getB());
        double expectedDim1 = -0.6700413426515754;
        double expectedDim2 = -0.6700413426515754;
        double expectedDim3 = -0.937479822012605;
        double expectedAve = -0.7591875024385853;

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);
        Helpers.testModelProtoSerialization(llModel, Regressor.class, p.getB());

        p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, true);
        llModel = t.train(p.getA());
        llEval = e.evaluate(llModel,p.getB());

        assertEquals(expectedDim1,llEval.r2(new Regressor(RegressionDataGenerator.firstDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim2,llEval.r2(new Regressor(RegressionDataGenerator.secondDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedDim3,llEval.r2(new Regressor(RegressionDataGenerator.thirdDimensionName,Double.NaN)),1e-6);
        assertEquals(expectedAve,llEval.averageR2(),1e-6);
    }


    @Test
    public void loadProtobufModel() throws IOException, URISyntaxException {
        Path path = Paths.get(TestCARTRegressionTrainer.class.getResource("cart-reg-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            ModelProto proto = ModelProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            SparseModel<Regressor> deserModel = (SparseModel<Regressor>) Model.deserialize(proto);

            assertEquals("4.3.1", deserModel.getProvenance().getTribuoVersion());

            Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);

            List<Prediction<Regressor>> deserOutput = deserModel.predict(p.getB());

            CARTRegressionTrainer trainer = new CARTRegressionTrainer();
            TreeModel<Regressor> model = trainer.train(p.getA());
            List<Prediction<Regressor>> output = model.predict(p.getB());

            assertEquals(deserOutput.size(), p.getB().size());
            assertTrue(Helpers.predictionListDistributionEquals(deserOutput, output));
        }
    }

    /**
     * Test protobuf generation method.
     * @throws IOException If the write failed.
     */
    public void generateModel() throws IOException {
        Pair<Dataset<Regressor>, Dataset<Regressor>> p = RegressionDataGenerator.threeDimDenseTrainTest(1.0, false);
        CARTRegressionTrainer trainer = new CARTRegressionTrainer();
        TreeModel<Regressor> model = trainer.train(p.getA());
        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","regression","rtree","cart-reg-431.tribuo"));
    }
}
