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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.datasource.LibSVMDataSource;
import org.junit.jupiter.api.Test;
import org.tribuo.interop.onnx.protos.ExampleTransformerProto;
import org.tribuo.interop.onnx.protos.OutputTransformerProto;
import org.tribuo.protos.ProtoUtil;
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
import java.util.stream.Stream;

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
            sessionOptions.setInterOpNumThreads(1);
            sessionOptions.setIntraOpNumThreads(1);

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

            Helpers.testModelProtoSerialization(transposedMNISTLR, Label.class, new MutableDataset<>(transposedMNIST), 1e-6);
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

    @ParameterizedTest
    @MethodSource("load431ExampleProtobufs")
    public void testExampleProto(String name, ExampleTransformer actualTransformer) throws URISyntaxException, IOException {
        Path distancePath = Paths.get(TestOnnxRuntime.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(distancePath)) {
            ExampleTransformerProto proto = ExampleTransformerProto.parseFrom(fis);
            ExampleTransformer distance = ProtoUtil.deserialize(proto);
            assertEquals(actualTransformer, distance);
        }
    }

    private static Stream<Arguments> load431ExampleProtobufs() {
        return Stream.of(
                Arguments.of("dense-431.tribuo", new DenseTransformer()),
                Arguments.of("image-431.tribuo", new ImageTransformer(3,64,64))
        );
    }

    @ParameterizedTest
    @MethodSource("load431OutputProtobufs")
    public void testOutputProto(String name, OutputTransformer<?> actualTransformer) throws URISyntaxException, IOException {
        Path distancePath = Paths.get(TestOnnxRuntime.class.getResource(name).toURI());
        try (InputStream fis = Files.newInputStream(distancePath)) {
            OutputTransformerProto proto = OutputTransformerProto.parseFrom(fis);
            OutputTransformer<?> distance = ProtoUtil.deserialize(proto);
            assertEquals(actualTransformer, distance);
        }
    }

    private static Stream<Arguments> load431OutputProtobufs() {
        return Stream.of(
                Arguments.of("lovo-431.tribuo", new LabelOneVOneTransformer()),
                Arguments.of("label-431.tribuo", new LabelTransformer()),
                Arguments.of("multilabel-431.tribuo", new MultiLabelTransformer()),
                Arguments.of("regressor-431.tribuo", new RegressorTransformer())
        );
    }

    public void generateProtobufs() throws IOException {
        Helpers.writeProtobuf(new DenseTransformer(), Paths.get("src","test","resources","org","tribuo","interop","onnx","dense-431.tribuo"));
        Helpers.writeProtobuf(new ImageTransformer(3,64,64), Paths.get("src","test","resources","org","tribuo","interop","onnx","image-431.tribuo"));
        Helpers.writeProtobuf(new LabelOneVOneTransformer(), Paths.get("src","test","resources","org","tribuo","interop","onnx","lovo-431.tribuo"));
        Helpers.writeProtobuf(new LabelTransformer(), Paths.get("src","test","resources","org","tribuo","interop","onnx","label-431.tribuo"));
        Helpers.writeProtobuf(new MultiLabelTransformer(), Paths.get("src","test","resources","org","tribuo","interop","onnx","multilabel-431.tribuo"));
        Helpers.writeProtobuf(new RegressorTransformer(), Paths.get("src","test","resources","org","tribuo","interop","onnx","regressor-431.tribuo"));
    }

}
