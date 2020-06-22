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

package org.tribuo.transform;

import org.tribuo.Dataset;
import org.tribuo.FeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.RealInfo;
import org.tribuo.impl.ArrayExample;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.transform.transformations.MeanStdDevTransformation;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class MeanStdDevTest {

    public static MutableDataset<MockOutput> generateDenseDataset(int seed) {
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
    public void testMeanZeroStdDevOne() {
        TransformationMap t = new TransformationMap(Collections.singletonList(new MeanStdDevTransformation()),new HashMap<>());
        testMeanStdDev(t,0.0,1.0);
    }

    @Test
    public void testMeanZeroStdDevFive() {
        TransformationMap t = new TransformationMap(Collections.singletonList(new MeanStdDevTransformation(0.0, 5.0)),new HashMap<>());
        testMeanStdDev(t,0.0,5.0);
    }

    @Test
    public void testMeanFiveStdDevOne() {
        TransformationMap t = new TransformationMap(Collections.singletonList(new MeanStdDevTransformation(5.0, 1.0)),new HashMap<>());
        testMeanStdDev(t,5.0,1.0);
    }

    @Test
    public void testMeanMinusFiveStdDevFive() {
        TransformationMap t = new TransformationMap(Collections.singletonList(new MeanStdDevTransformation(-5.0, 5.0)),new HashMap<>());
        testMeanStdDev(t,-5.0,5.0);
    }

    @Test
    public void testInvalidTransformation() {
        try {
            Transformation t = new MeanStdDevTransformation(0.0, 0.0);
            fail("Should have thrown exception");
        } catch (IllegalArgumentException e) {

        } catch (Exception e) {
            fail("Threw incorrect exception, should have been IllegalArgumentException, found " + e);
        }
        try {
            Transformation t = new MeanStdDevTransformation(0.0, -1.0);
            fail("Should have thrown exception");
        } catch (IllegalArgumentException e) {

        } catch (Exception e) {
            fail("Threw incorrect exception, should have been IllegalArgumentException, found " + e);
        }
    }

    public void testMeanStdDev(TransformationMap t, double targetMean, double targetStdDev) {
        MutableDataset<MockOutput> train = generateDenseDataset(1);
        MutableDataset<MockOutput> test = generateDenseDataset(2);

        TransformerMap tMap = train.createTransformers(t);

        Dataset<MockOutput> transformedTrain = tMap.transformDataset(train);
        Dataset<MockOutput> transformedTest = tMap.transformDataset(test);

        FeatureMap trainFMap = transformedTrain.getFeatureMap();
        FeatureMap testFMap = transformedTest.getFeatureMap();

        assertEquals(targetMean,((RealInfo)trainFMap.get("F0")).getMean(),1e-5);
        assertEquals(targetStdDev,Math.sqrt(((RealInfo)trainFMap.get("F0")).getVariance()),1e-5);

        assertEquals(targetMean,((RealInfo)trainFMap.get("F1")).getMean(),1e-5);
        assertEquals(targetStdDev,Math.sqrt(((RealInfo)trainFMap.get("F1")).getVariance()),1e-5);

        assertEquals(targetMean,((RealInfo)testFMap.get("F0")).getMean(),1e-1);
        assertEquals(targetStdDev,Math.sqrt(((RealInfo)testFMap.get("F0")).getVariance()),1e-1);

        assertEquals(targetMean,((RealInfo)testFMap.get("F1")).getMean(),1e-2);
        assertEquals(targetStdDev,Math.sqrt(((RealInfo)testFMap.get("F1")).getVariance()),1e-2);
    }

}
