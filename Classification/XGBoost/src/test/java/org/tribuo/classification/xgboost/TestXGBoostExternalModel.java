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

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.common.xgboost.XGBoostExternalModel;
import org.tribuo.datasource.LibSVMDataSource;
import org.junit.jupiter.api.Test;
import org.tribuo.test.Helpers;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestXGBoostExternalModel {
    @Test
    public void testMNIST() throws IOException, URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();
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

        Helpers.testModelSerialization(transposedMNISTXGB,Label.class);
        Helpers.testModelProtoSerialization(transposedMNISTXGB, Label.class, transposedMNIST);
    }

    @Test
    public void testTransposedMNIST() throws IOException, URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();
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
}
