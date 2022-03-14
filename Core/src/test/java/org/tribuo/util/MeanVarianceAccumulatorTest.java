/*
 * Copyright (c) 2020, 2022, Oracle and/or its affiliates. All rights reserved.
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
package org.tribuo.util;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.Trainer;

import java.util.SplittableRandom;

public class MeanVarianceAccumulatorTest {

    private static final double DELTA = 1e-14;

    @Test
    public void testDifferentMethods() {
        double[] values = new double[] {1, -2, 3, -4, 5, -5, 4, -3, 2, -1};

        MeanVarianceAccumulator zeroed = new MeanVarianceAccumulator();
        for (int i = 0; i < values.length; i++) {
            zeroed.observe(values[i]);
        }

        MeanVarianceAccumulator zeroedObserveArray = new MeanVarianceAccumulator();
        zeroedObserveArray.observe(values);

        MeanVarianceAccumulator observed = new MeanVarianceAccumulator(values);

        // Test ordering
        Assertions.assertEquals(observed,zeroed);
        Assertions.assertEquals(zeroed,zeroedObserveArray);

        // Test outputs
        Assertions.assertEquals(0.0,zeroed.getMean(), DELTA);
        Assertions.assertEquals(5,zeroed.getMax(), DELTA);
        Assertions.assertEquals(-5,zeroed.getMin(), DELTA);
    }

    @Test
    public void testDifferentOrder() {
        double[] values = new double[] {1, -2, 3, -4, 5, -5, 4, -3, 2, -1};

        MeanVarianceAccumulator base = new MeanVarianceAccumulator(values);

        SplittableRandom rng = new SplittableRandom(Trainer.DEFAULT_SEED);

        for (int i = 0; i < 100; i++) {
            Util.randpermInPlace(values,rng);
            MeanVarianceAccumulator shuffled = new MeanVarianceAccumulator(values);
            Assertions.assertEquals(base.getMin(),shuffled.getMin(),DELTA);
            Assertions.assertEquals(base.getMax(),shuffled.getMax(),DELTA);
            Assertions.assertEquals(base.getMean(),shuffled.getMean(),DELTA);
            Assertions.assertEquals(base.getVariance(),shuffled.getVariance(),DELTA);
            Assertions.assertEquals(base.getCount(),shuffled.getCount(),DELTA);
        }
    }

    @Test
    public void testRandomized() {
        SplittableRandom rng = new SplittableRandom(Trainer.DEFAULT_SEED);

        double[] values = new double[4096];

        for (int i = 0; i < values.length; i++) {
            values[i] = rng.nextDouble();
        }

        MeanVarianceAccumulator accumulator = new MeanVarianceAccumulator(values);

        double mean = Util.mean(values);
        Assertions.assertEquals(mean,accumulator.getMean(), DELTA);

        Pair<Double, Double> meanVarPair = Util.meanAndVariance(values);
        Assertions.assertEquals(meanVarPair.getA(),accumulator.getMean(), DELTA);
        Assertions.assertEquals(meanVarPair.getB(),accumulator.getVariance(), DELTA);
    }

    @Test
    public void testMerge() {
        SplittableRandom rng = new SplittableRandom(Trainer.DEFAULT_SEED);

        double[] values = new double[4096];

        for (int i = 0; i < values.length; i++) {
            values[i] = rng.nextDouble();
        }

        double[] firstValues = new double[3072];
        double[] secondValues = new double[1024];

        System.arraycopy(values, 0, firstValues, 0, firstValues.length);
        System.arraycopy(values, 3072, secondValues, 0, secondValues.length);

        MeanVarianceAccumulator totalAccumulator = new MeanVarianceAccumulator(values);

        MeanVarianceAccumulator first = new MeanVarianceAccumulator(firstValues);
        MeanVarianceAccumulator second = new MeanVarianceAccumulator(secondValues);

        MeanVarianceAccumulator sum = MeanVarianceAccumulator.merge(first,second);

        Assertions.assertEquals(totalAccumulator.getMin(),sum.getMin(), DELTA);
        Assertions.assertEquals(totalAccumulator.getMax(),sum.getMax(), DELTA);
        Assertions.assertEquals(totalAccumulator.getMean(),sum.getMean(), DELTA);
        Assertions.assertEquals(totalAccumulator.getVariance(),sum.getVariance(), DELTA);
        Assertions.assertEquals(totalAccumulator.getCount(),sum.getCount());
    }

}
