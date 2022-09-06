/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.MutableDataset;
import org.tribuo.interop.tensorflow.example.GraphDefTuple;
import org.tribuo.interop.tensorflow.example.MLPExamples;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.evaluation.RegressionEvaluation;
import org.tribuo.regression.evaluation.RegressionEvaluator;
import org.tribuo.regression.example.NonlinearGaussianDataSource;
import org.tribuo.test.Helpers;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class RegressionTest {

    private static final String INPUT_NAME = "input";

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(TensorFlowTrainer.class.getName());
        logger.setLevel(Level.WARNING);
    }

    @Test
    public void regressionMLPTest() throws IOException {
        // Create the train and test data
        DataSource<Regressor> trainSource = new NonlinearGaussianDataSource(1024,new float[]{1.0f,2.0f,-3.0f,4.0f},1.0f,0.1f,-5.0f,5.0f,-1.0f,1.0f,42);
        DataSource<Regressor> testSource = new NonlinearGaussianDataSource(1024,new float[]{1.0f,2.0f,-3.0f,4.0f},1.0f,0.1f,-5.0f,5.0f,-1.0f,1.0f,42*42);
        MutableDataset<Regressor> trainData = new MutableDataset<>(trainSource);
        MutableDataset<Regressor> testData = new MutableDataset<>(testSource);

        // Build the MLP graph
        GraphDefTuple graphDefTuple = MLPExamples.buildMLPGraph(INPUT_NAME, trainData.getFeatureMap().size(), new int[]{50,50}, trainData.getOutputs().size());

        // Configure the trainer
        Map<String, Float> gradientParams = new HashMap<>();
        gradientParams.put("learningRate", 0.01f);
        gradientParams.put("initialAccumulatorValue", 0.1f);
        FeatureConverter denseConverter = new DenseFeatureConverter(INPUT_NAME);
        OutputConverter<Regressor> outputConverter = new RegressorConverter();
        TensorFlowTrainer<Regressor> trainer = new TensorFlowTrainer<>(graphDefTuple.graphDef,
                graphDefTuple.outputName,
                GradientOptimiser.ADAGRAD,
                gradientParams,
                denseConverter,
                outputConverter,
                16,
                10,
                16,
                -1);

        // Train the model
        TensorFlowModel<Regressor> model = trainer.train(trainData);

        // Run smoke test evaluation
        RegressionEvaluation eval = new RegressionEvaluator().evaluate(model,testData);
        Assertions.assertFalse(eval.r2().isEmpty());

        // Check Tribuo serialization
        Helpers.testModelSerialization(model,Regressor.class);
        Helpers.testModelProtoSerialization(model, Regressor.class, testData);

        // Check saved model bundle export
        Path outputPath = Files.createTempDirectory("tf-regression-test");
        model.exportModel(outputPath.toString());
        List<Path> files = Files.list(outputPath).collect(Collectors.toList());
        Assertions.assertNotEquals(0,files.size());

        // Cleanup saved model bundle
        Files.walk(outputPath)
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);

        Assertions.assertFalse(Files.exists(outputPath));

        // Cleanup created model
        model.close();
    }

}
