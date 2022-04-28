/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.transform.transformations;

import org.tribuo.CategoricalInfo;
import org.tribuo.Dataset;
import org.tribuo.FeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.TransformerProto;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.Transformer;
import org.tribuo.transform.TransformerMap;
import org.tribuo.transform.transformations.BinningTransformation.BinningTransformer;
import org.tribuo.transform.transformations.BinningTransformation.BinningType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 *
 */
public class BinningTest {

    public static MutableDataset<MockOutput> generateRandomDenseDataset(int seed) {
        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(),new MockOutputFactory());

        Random rng = new Random(seed);
        MockOutput output = new MockOutput("UNK");
        String[] featureNames = new String[]{"F0","F1"};

        for (int i = 0; i < 10000; i++) {
            double f0 = (rng.nextGaussian() * 5) + 10;
            double f1 = rng.nextDouble() * -20;
            ArrayExample<MockOutput> example = new ArrayExample<>(output,featureNames,new double[]{f0,f1});
            dataset.add(example);
        }

        return dataset;
    }

    @Test
    public void testEqualWidthBinning() {
        TransformationMap t = new TransformationMap(Collections.singletonList(BinningTransformation.equalWidth(5)),new HashMap<>());
        MutableDataset<MockOutput> train = generateRandomDenseDataset(1);
        MutableDataset<MockOutput> test = generateRandomDenseDataset(2);

        TransformerMap tMap = train.createTransformers(t);

        Dataset<MockOutput> transformedTrain = tMap.transformDataset(train);
        Dataset<MockOutput> transformedTest = tMap.transformDataset(test);

        FeatureMap trainFMap = transformedTrain.getFeatureMap();
        FeatureMap testFMap = transformedTest.getFeatureMap();

        assertEquals(140,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(1));
        assertEquals(2349,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(2));
        assertEquals(5463,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(3));
        assertEquals(1946,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(4));
        assertEquals(102,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(5));

        assertEquals(2015,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(1));
        assertEquals(1985,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(2));
        assertEquals(2034,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(3));
        assertEquals(1979,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(4));
        assertEquals(1987,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(5));

        assertEquals(154,((CategoricalInfo)testFMap.get("F0")).getObservationCount(1));
        assertEquals(2364,((CategoricalInfo)testFMap.get("F0")).getObservationCount(2));
        assertEquals(5523,((CategoricalInfo)testFMap.get("F0")).getObservationCount(3));
        assertEquals(1867,((CategoricalInfo)testFMap.get("F0")).getObservationCount(4));
        assertEquals(92,((CategoricalInfo)testFMap.get("F0")).getObservationCount(5));

        assertEquals(1976,((CategoricalInfo)testFMap.get("F1")).getObservationCount(1));
        assertEquals(2004,((CategoricalInfo)testFMap.get("F1")).getObservationCount(2));
        assertEquals(2033,((CategoricalInfo)testFMap.get("F1")).getObservationCount(3));
        assertEquals(1980,((CategoricalInfo)testFMap.get("F1")).getObservationCount(4));
        assertEquals(2007,((CategoricalInfo)testFMap.get("F1")).getObservationCount(5));

        List<Transformer> f0Transformer = tMap.get("F0");
        assertEquals(1,f0Transformer.size());

        TransformerProto proto = f0Transformer.get(0).serialize();
        Transformer transformer = Transformer.deserialize(proto);

        assertEquals(f0Transformer.get(0), transformer);
        assertNotSame(f0Transformer.get(0), transformer);
    }

    @Test
    public void testMediansBinning() {
        TransformationMap t = new TransformationMap(Collections.singletonList(BinningTransformation.equalFrequency(5)),new HashMap<>());
        MutableDataset<MockOutput> train = generateRandomDenseDataset(1);
        MutableDataset<MockOutput> test = generateRandomDenseDataset(2);

        TransformerMap tMap = train.createTransformers(t);

        Dataset<MockOutput> transformedTrain = tMap.transformDataset(train);
        Dataset<MockOutput> transformedTest = tMap.transformDataset(test);

        FeatureMap trainFMap = transformedTrain.getFeatureMap();
        FeatureMap testFMap = transformedTest.getFeatureMap();

        assertEquals(2001,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(1));
        assertEquals(2000,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(2));
        assertEquals(2000,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(3));
        assertEquals(2000,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(4));
        assertEquals(1999,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(5));

        assertEquals(2001,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(1));
        assertEquals(2000,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(2));
        assertEquals(2000,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(3));
        assertEquals(2000,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(4));
        assertEquals(1999,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(5));

        assertEquals(2024,((CategoricalInfo)testFMap.get("F0")).getObservationCount(1));
        assertEquals(2061,((CategoricalInfo)testFMap.get("F0")).getObservationCount(2));
        assertEquals(2038,((CategoricalInfo)testFMap.get("F0")).getObservationCount(3));
        assertEquals(1955,((CategoricalInfo)testFMap.get("F0")).getObservationCount(4));
        assertEquals(1922,((CategoricalInfo)testFMap.get("F0")).getObservationCount(5));

        assertEquals(1962,((CategoricalInfo)testFMap.get("F1")).getObservationCount(1));
        assertEquals(2019,((CategoricalInfo)testFMap.get("F1")).getObservationCount(2));
        assertEquals(2001,((CategoricalInfo)testFMap.get("F1")).getObservationCount(3));
        assertEquals(1995,((CategoricalInfo)testFMap.get("F1")).getObservationCount(4));
        assertEquals(2023,((CategoricalInfo)testFMap.get("F1")).getObservationCount(5));
    }

    @Test
    public void testStdDevBinning() {
        TransformationMap t = new TransformationMap(Collections.singletonList(BinningTransformation.stdDevs(3)),new HashMap<>());
        MutableDataset<MockOutput> train = generateRandomDenseDataset(1);
        MutableDataset<MockOutput> test = generateRandomDenseDataset(2);

        TransformerMap tMap = train.createTransformers(t);

        Dataset<MockOutput> transformedTrain = tMap.transformDataset(train);
        Dataset<MockOutput> transformedTest = tMap.transformDataset(test);

        FeatureMap trainFMap = transformedTrain.getFeatureMap();
        FeatureMap testFMap = transformedTest.getFeatureMap();

        assertEquals(221,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(1));
        assertEquals(1383,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(2));
        assertEquals(3391,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(3));
        assertEquals(3415,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(4));
        assertEquals(1365,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(5));
        assertEquals(225,((CategoricalInfo)trainFMap.get("F0")).getObservationCount(6));

        assertEquals(0,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(1));
        assertEquals(2142,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(2));
        assertEquals(2863,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(3));
        assertEquals(2889,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(4));
        assertEquals(2106,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(5));
        assertEquals(0,((CategoricalInfo)trainFMap.get("F1")).getObservationCount(6));

        assertEquals(222,((CategoricalInfo)testFMap.get("F0")).getObservationCount(1));
        assertEquals(1382,((CategoricalInfo)testFMap.get("F0")).getObservationCount(2));
        assertEquals(3482,((CategoricalInfo)testFMap.get("F0")).getObservationCount(3));
        assertEquals(3399,((CategoricalInfo)testFMap.get("F0")).getObservationCount(4));
        assertEquals(1294,((CategoricalInfo)testFMap.get("F0")).getObservationCount(5));
        assertEquals(221,((CategoricalInfo)testFMap.get("F0")).getObservationCount(6));

        assertEquals(0,((CategoricalInfo)testFMap.get("F1")).getObservationCount(1));
        assertEquals(2102,((CategoricalInfo)testFMap.get("F1")).getObservationCount(2));
        assertEquals(2917,((CategoricalInfo)testFMap.get("F1")).getObservationCount(3));
        assertEquals(2868,((CategoricalInfo)testFMap.get("F1")).getObservationCount(4));
        assertEquals(2113,((CategoricalInfo)testFMap.get("F1")).getObservationCount(5));
        assertEquals(0,((CategoricalInfo)testFMap.get("F1")).getObservationCount(6));
    }

    @Test
    void testSerializeBinningTransformer() throws Exception {
        BinningTransformer bt = new BinningTransformer(BinningType.EQUAL_FREQUENCY, new double[] {0.0, 0.1, 0.2}, new double[] {1.0, 10.0, 100.0});
        TransformerProto tp = bt.serialize();
        BinningTransformer btd = ProtoUtil.deserialize(tp);
        assertEquals(bt, btd);
    }
}
