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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
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

public class TestOnnxRuntime {

    /**
     * This test checks that the ImageTransformer works and we can process float matrices through the LabelTransformer.
     * @throws IOException If it failed to read the file.
     * @throws OrtException If onnx-runtime failed.
     * @throws URISyntaxException If the URL failed to parse.
     */
    @Test
    public void testCNNMNIST() throws IOException, OrtException, URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

            // Loads regular MNIST
            URL data = TestOnnxRuntime.class.getResource("/org/tribuo/interop/onnx/mnist_test_head.libsvm");
            DataSource<Label> mnistTest = new LibSVMDataSource<>(data, labelFactory, false, 784);
            Dataset<Label> dataset = new MutableDataset<>(mnistTest);

            Map<String, Integer> featureMapping = new HashMap<>();
            for (int i = 0; i < 784; i++) {
                featureMapping.put(String.format("%03d", i), i);
            }
            Map<Label, Integer> outputMapping = new HashMap<>();
            for (Label l : dataset.getOutputInfo().getDomain()) {
                outputMapping.put(l, Integer.parseInt(l.getLabel()));
            }

            ImageTransformer imageTransformer = new ImageTransformer(1,28,28);
            LabelTransformer labelTransformer = new LabelTransformer();
            Path testResource = Paths.get(TestOnnxRuntime.class.getResource("/org/tribuo/interop/onnx/cnn_mnist.onnx").toURI());
            ONNXExternalModel<Label> cnnModel = ONNXExternalModel.createOnnxModel(
                    labelFactory, featureMapping, outputMapping, imageTransformer,
                    labelTransformer, sessionOptions, testResource, "input_image");

            LabelEvaluation evaluation = labelFactory.getEvaluator().evaluate(cnnModel, mnistTest);
            // CNNs are good at MNIST
            assertEquals(1.0, evaluation.accuracy(), 1e-6);
            assertEquals(0.0, evaluation.balancedErrorRate(), 1e-6);
        }
    }

    /**
     * This test checks that the model works when using the feature mapping logic as the model was trained with
     * a transposed feature mapping, but the data is loaded in using the standard mapping.
     * @throws IOException If it failed to read the file.
     * @throws OrtException If onnx-runtime failed.
     * @throws URISyntaxException If the URL failed to parse.
     */
    @Test
    public void testMNIST() throws IOException, OrtException, URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

            // Loads regular MNIST
            URL data = TestOnnxRuntime.class.getResource("/org/tribuo/interop/onnx/mnist_test_head.libsvm");
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

            DenseTransformer denseTransformer = new DenseTransformer();
            LabelTransformer labelTransformer = new LabelTransformer();
            Path testResource = Paths.get(TestOnnxRuntime.class.getResource("/org/tribuo/interop/onnx/lr_mnist.onnx").toURI());
            ONNXExternalModel<Label> transposedMNISTLR = ONNXExternalModel.createOnnxModel(
                    labelFactory, featureMapping, outputMapping, denseTransformer,
                    labelTransformer, sessionOptions, testResource, "float_input");
            // This model doesn't have a free batch size parameter on the output
            transposedMNISTLR.setBatchSize(1);

            LabelEvaluation evaluation = labelFactory.getEvaluator().evaluate(transposedMNISTLR, transposedMNIST);
            assertEquals(0.967741, evaluation.accuracy(), 1e-6);
            assertEquals(0.024285, evaluation.balancedErrorRate(), 1e-6);

            Helpers.testModelSerialization(transposedMNISTLR,Label.class);
        }
    }

    /**
     * This test checks that the model works with the identity feature mapping when the data is transposed.
     * @throws IOException If it failed to read the file.
     * @throws OrtException If onnx-runtime failed.
     * @throws URISyntaxException If the URL failed to parse.
     */
    @Test
    public void testTransposedMNIST() throws IOException, OrtException, URISyntaxException {
        LabelFactory labelFactory = new LabelFactory();
        try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();

            // Loads transposed MNIST
            URL data = TestOnnxRuntime.class.getResource("/org/tribuo/interop/onnx/transposed_mnist_test_head.libsvm");
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

            DenseTransformer denseTransformer = new DenseTransformer();
            LabelTransformer labelTransformer = new LabelTransformer();
            Path testResource = Paths.get(TestOnnxRuntime.class.getResource("/org/tribuo/interop/onnx/lr_mnist.onnx").toURI());
            ONNXExternalModel<Label> transposedMNISTLR = ONNXExternalModel.createOnnxModel(
                    labelFactory, featureMapping, outputMapping, denseTransformer,
                    labelTransformer, sessionOptions, testResource, "float_input");
            // This model doesn't have a free batch size parameter on the output
            transposedMNISTLR.setBatchSize(1);

            LabelEvaluation evaluation = labelFactory.getEvaluator().evaluate(transposedMNISTLR, transposedMNIST);
            assertEquals(0.967741, evaluation.accuracy(), 1e-6);
            assertEquals(0.024285, evaluation.balancedErrorRate(), 1e-6);
        }
    }

}
