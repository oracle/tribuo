/*
 * Copyright (c) 2015, 2023, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.common.xgboost.XGBoostExternalModel;
import org.tribuo.datasource.LibSVMDataSource;
import org.junit.jupiter.api.Test;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestXGBoostExternalModel {
    private static final LabelFactory labelFactory = new LabelFactory();

    @Test
    public void testMNIST() throws IOException, URISyntaxException {
        // Loads regular MNIST
        URL data = TestXGBoostExternalModel.class.getResource("/org/tribuo/classification/xgboost/mnist_test_head.libsvm");
        DataSource<Label> transposedMNIST = new LibSVMDataSource<>(data, labelFactory, false, 784);
        Dataset<Label> dataset = new MutableDataset<>(transposedMNIST);

        Map<String, Integer> featureMapping = new HashMap<>();
        for (int i = 0; i < 784; i++) {
            // This MNIST model has the feature indices transposed to test a non-trivial mapping.
            int id = (783 - i);
            featureMapping.put(String.format("%03d", i), id);
        }
        Map<Label, Integer> outputMapping = new HashMap<>();
        for (Label l : dataset.getOutputInfo().getDomain()) {
            outputMapping.put(l, Integer.parseInt(l.getLabel()));
        }

        XGBoostClassificationConverter labelConverter = new XGBoostClassificationConverter();
        Path testResource = Paths.get(TestXGBoostExternalModel.class.getResource("/org/tribuo/classification/xgboost/xgb_mnist.xgb").toURI());
        XGBoostExternalModel<Label> transposedMNISTXGB = XGBoostExternalModel.createXGBoostModel(
                labelFactory, featureMapping, outputMapping, labelConverter, testResource);

        LabelEvaluation evaluation = labelFactory.getEvaluator().evaluate(transposedMNISTXGB, transposedMNIST);
        assertEquals(1.0, evaluation.accuracy(), 1e-6);
        assertEquals(0.0, evaluation.balancedErrorRate(), 1e-6);

        Helpers.testModelProtoSerialization(transposedMNISTXGB, Label.class, transposedMNIST);
    }

    @Test
    public void testTransposedMNIST() throws IOException, URISyntaxException {
        // Loads transposed MNIST
        URL data = XGBoostExternalModel.class.getResource("/org/tribuo/classification/xgboost/transposed_mnist_test_head.libsvm");
        DataSource<Label> transposedMNIST = new LibSVMDataSource<>(data, labelFactory, true, 784);
        Dataset<Label> dataset = new MutableDataset<>(transposedMNIST);

        Map<String, Integer> featureMapping = new HashMap<>();
        for (int i = 0; i < 784; i++) {
            featureMapping.put(String.format("%03d", i), i);
        }
        Map<Label, Integer> outputMapping = new HashMap<>();
        for (Label l : dataset.getOutputInfo().getDomain()) {
            outputMapping.put(l, Integer.parseInt(l.getLabel()));
        }

        XGBoostClassificationConverter labelConverter = new XGBoostClassificationConverter();
        Path testResource = Paths.get(TestXGBoostExternalModel.class.getResource("/org/tribuo/classification/xgboost/xgb_mnist.xgb").toURI());
        XGBoostExternalModel<Label> transposedMNISTXGB = XGBoostExternalModel.createXGBoostModel(
                labelFactory, featureMapping, outputMapping, labelConverter, testResource);

        LabelEvaluation evaluation = labelFactory.getEvaluator().evaluate(transposedMNISTXGB, transposedMNIST);
        assertEquals(1.0, evaluation.accuracy(), 1e-6);
        assertEquals(0.0, evaluation.balancedErrorRate(), 1e-6);
    }

    @Test
    public void loadProtobufModel() throws IOException, URISyntaxException {
        // Loads transposed MNIST
        URL data = XGBoostExternalModel.class.getResource("/org/tribuo/classification/xgboost/transposed_mnist_test_head.libsvm");
        DataSource<Label> transposedMNIST = new LibSVMDataSource<>(data, labelFactory, true, 784);

        Path path = Paths.get(TestXGBoost.class.getResource("xgboost-clf-ext-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(path)) {
            ModelProto proto = ModelProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            XGBoostExternalModel<Label> model = (XGBoostExternalModel<Label>) Model.deserialize(proto);

            assertEquals("4.3.1", model.getProvenance().getTribuoVersion());

            LabelEvaluation evaluation = labelFactory.getEvaluator().evaluate(model, transposedMNIST);
            assertEquals(1.0, evaluation.accuracy(), 1e-6);
            assertEquals(0.0, evaluation.balancedErrorRate(), 1e-6);
        }
    }

    public void generateModel() throws IOException, URISyntaxException {
        Map<String, Integer> featureMapping = new HashMap<>();
        for (int i = 0; i < 784; i++) {
            featureMapping.put(String.format("%03d", i), i);
        }
        Map<Label, Integer> outputMapping = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            outputMapping.put(new Label(String.valueOf(i)), i);
        }

        XGBoostClassificationConverter labelConverter = new XGBoostClassificationConverter();
        Path testResource = Paths.get(TestXGBoostExternalModel.class.getResource("/org/tribuo/classification/xgboost/xgb_mnist.xgb").toURI());
        XGBoostExternalModel<Label> model = XGBoostExternalModel.createXGBoostModel(
                labelFactory, featureMapping, outputMapping, labelConverter, testResource);
        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","classification","xgboost","xgboost-clf-ext-431.tribuo"));
    }
}
