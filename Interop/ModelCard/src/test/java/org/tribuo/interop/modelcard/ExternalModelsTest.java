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

package org.tribuo.interop.modelcard;

import org.junit.jupiter.api.Test;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.xgboost.XGBoostClassificationConverter;
import org.tribuo.common.xgboost.XGBoostExternalModel;
import org.tribuo.data.csv.CSVLoader;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.fail;

public class ExternalModelsTest {
    private static final Logger logger = Logger.getLogger(ExternalModelsTest.class.getName());

    @Test
    public void testExternalModelException() throws IOException {
        String arch = System.getProperty("os.arch");
        if (arch.equalsIgnoreCase("amd64") || arch.equalsIgnoreCase("x86_64")) {
            var labelFactory = new LabelFactory();
            var csvLoader = new CSVLoader<>(labelFactory);

            var dataSource = csvLoader.loadDataSource(Paths.get("src/test/resources/externalModelSampleData.csv"), "response");
            var trainData = new MutableDataset<>(dataSource);
            var evalData = new MutableDataset<>(dataSource);

            var xgbLabelConv = new XGBoostClassificationConverter();
            var xgbModelPath = Paths.get("src/test/resources/externalModelPath.xgb");

            Map<String, Integer> xgbFeatMapping = new HashMap<>();
            for (int i = 0; i < 784; i++) {
                int id = (783 - i);
                xgbFeatMapping.put(String.format("%03d", i), id);
            }
            Map<Label, Integer> xgbOutMapping = new HashMap<>();
            int i = 0;
            for (Label l : trainData.getOutputInfo().getDomain()) {
                xgbOutMapping.put(l, i);
                i++;
            }
            Model<Label> xgbModel = XGBoostExternalModel.createXGBoostModel(labelFactory, xgbFeatMapping, xgbOutMapping, xgbLabelConv, xgbModelPath);
            var xgbEvaluation = labelFactory.getEvaluator().evaluate(xgbModel,evalData);
            try {
                ModelCard modelCard = new ModelCard(xgbModel, xgbEvaluation, new UsageDetailsBuilder().build());
                fail("Exception expected");
            } catch (IllegalArgumentException e) {
                // test passed
            }
        } else {
            logger.warning("ORT based tests only supported on x86_64, found " + arch);
        }
    }
}