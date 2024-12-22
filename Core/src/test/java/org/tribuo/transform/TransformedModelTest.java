/*
 * Copyright (c) 2023, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.transform;

import org.junit.jupiter.api.Test;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.PredictionTest;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.PredictionProto;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.test.MockTrainer;
import org.tribuo.transform.transformations.BinningTransformation;
import org.tribuo.transform.transformations.IDFTransformation;
import org.tribuo.transform.transformations.LinearScalingTransformation;
import org.tribuo.transform.transformations.MeanStdDevTransformation;
import org.tribuo.transform.transformations.SimpleTransform;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TransformedModelTest {

    public static MutableDataset<MockOutput> generateDenseDataset() {
        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(), new MockOutputFactory());

        MockOutput output = new MockOutput("UNK");
        String[] featureNames = new String[]{"F0", "F1", "F2", "F3", "F4"};
        ArrayExample<MockOutput> example;

        example = new ArrayExample<>(output, featureNames, new double[]{1, 1, 1, 1, 1});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{1, 2, 3, 4, 5});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{0.5, 0.5, 0.5, 0.5, 0.5});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{0, 0, 0, 0, 0});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{10, 9, 8, 7, 6});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{2, 2, 2, 2, 2});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{10, 10, 10, 10, 10});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{1, 5, 1, 5, 1});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{5, 1, 5, 1, 5});
        dataset.add(example);

        example = new ArrayExample<>(output, featureNames, new double[]{1, 2, 3, 4, 5});
        dataset.add(example);

        return dataset;
    }

    @Test
    public void serializationTest() {
        MutableDataset<MockOutput> dataset = generateDenseDataset();
        TransformationMap transformationMap = new TransformationMap(Collections.singletonList(new LinearScalingTransformation()));
        MockTrainer trainer = new MockTrainer("UNK");
        TransformTrainer<MockOutput> transformTrainer = new TransformTrainer<>(trainer, transformationMap);

        TransformedModel<MockOutput> model = transformTrainer.train(dataset);

        Helpers.testModelProtoSerialization(model, MockOutput.class, dataset);
    }

    @Test
    public void test431Protobufs() throws IOException, URISyntaxException {
        // Recreate model
        MutableDataset<MockOutput> dataset = generateDenseDataset();
        HashMap<String, List<Transformation>> map = new HashMap<>();
        map.put("F0", Collections.singletonList(BinningTransformation.equalWidth(3)));
        map.put("F1", Collections.singletonList(new IDFTransformation()));
        map.put("F2", Collections.singletonList(new LinearScalingTransformation(1,10)));
        map.put("F3", Collections.singletonList(new MeanStdDevTransformation(1, 5)));
        map.put("F4", Collections.singletonList(SimpleTransform.mul(5)));
        TransformationMap transformationMap = new TransformationMap(map);
        MockTrainer trainer = new MockTrainer("UNK");
        TransformTrainer<MockOutput> transformTrainer = new TransformTrainer<>(trainer, transformationMap);
        TransformedModel<MockOutput> model = transformTrainer.train(dataset);

        Path modelPath = Paths.get(TransformedModelTest.class.getResource("transformed-model-431.tribuo").toURI());
        try (InputStream fis = Files.newInputStream(modelPath)) {
            ModelProto proto = ModelProto.parseFrom(fis);
            @SuppressWarnings("unchecked")
            TransformedModel<MockOutput> deserModel = (TransformedModel<MockOutput>) Model.deserialize(proto);
            assertEquals("4.3.1", deserModel.getProvenance().getTribuoVersion());

            // As it's a mocked model underneath we run basic machinery tests to ensure the model deserialized, but
            // don't check the output as it is gibberish.
            List<Prediction<MockOutput>> output = model.predict(dataset);
            List<Prediction<MockOutput>> deserOutput = deserModel.predict(dataset);
            assertEquals(output.size(), deserOutput.size());
            assertEquals(dataset.size(), deserOutput.size());
        }
    }

    public void generateProtobufs() throws IOException {
        MutableDataset<MockOutput> dataset = generateDenseDataset();
        HashMap<String, List<Transformation>> map = new HashMap<>();
        map.put("F0", Collections.singletonList(BinningTransformation.equalWidth(3)));
        map.put("F1", Collections.singletonList(new IDFTransformation()));
        map.put("F2", Collections.singletonList(new LinearScalingTransformation(1,10)));
        map.put("F3", Collections.singletonList(new MeanStdDevTransformation(1, 5)));
        map.put("F4", Collections.singletonList(SimpleTransform.mul(5)));
        TransformationMap transformationMap = new TransformationMap(map);
        MockTrainer trainer = new MockTrainer("UNK");
        TransformTrainer<MockOutput> transformTrainer = new TransformTrainer<>(trainer, transformationMap);
        TransformedModel<MockOutput> model = transformTrainer.train(dataset);

        Helpers.writeProtobuf(model, Paths.get("src","test","resources","org","tribuo","transform","transformed-model-431.tribuo"));
    }
}
