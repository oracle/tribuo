/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.test.MockTrainer;
import org.tribuo.transform.transformations.LinearScalingTransformation;

import java.util.Collections;

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
}
