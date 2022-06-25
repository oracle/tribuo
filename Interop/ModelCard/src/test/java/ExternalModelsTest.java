/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Test;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.xgboost.XGBoostClassificationConverter;
import org.tribuo.common.xgboost.XGBoostExternalModel;
import org.tribuo.datasource.IDXDataSource;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.fail;

public class ExternalModelsTest {
    @Test
    public void testExternalModelException() throws IOException {
        var labelFactory = new LabelFactory();
        var mnistTrainSource = new IDXDataSource<>(Paths.get("src/test/input-data/train-images-idx3-ubyte.gz"),Paths.get("src/test/input-data/train-labels-idx1-ubyte.gz"),labelFactory);
        var mnistTestSource = new IDXDataSource<>(Paths.get("src/test/input-data/t10k-images-idx3-ubyte.gz"),Paths.get("src/test/input-data/t10k-labels-idx1-ubyte.gz"),labelFactory);
        var mnistTrain = new MutableDataset<>(mnistTrainSource);
        var mnistTest = new MutableDataset<>(mnistTestSource);

        var xgbLabelConv = new XGBoostClassificationConverter();
        var xgbModelPath = Paths.get("src/test/input-data/xgb_mnist.xgb");

        Map<String, Integer> xgbFeatMapping = new HashMap<>();
        for (int i = 0; i < 784; i++) {
            // This MNIST model has the feature indices transposed to test a non-trivial mapping.
            int id = (783 - i);
            xgbFeatMapping.put(String.format("%03d", i), id);
        }
        Map<Label, Integer> xgbOutMapping = new HashMap<>();
        for (Label l : mnistTrain.getOutputInfo().getDomain()) {
            xgbOutMapping.put(l, Integer.parseInt(l.getLabel()));
        }
        Model<Label> xgbModel = XGBoostExternalModel.createXGBoostModel(labelFactory, xgbFeatMapping, xgbOutMapping, xgbLabelConv, xgbModelPath);
        var xgbEvaluation = labelFactory.getEvaluator().evaluate(xgbModel,mnistTest);

        try {
            ModelCard modelCard = new ModelCard(xgbModel, xgbEvaluation);
            fail("Exception expected");
        } catch (IllegalArgumentException e) {
            // test passed
        }
    }
}