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

package org.tribuo;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class CategoricalInfoTest {

    private static final double DELTA = 1e-12;

    private static final int NUM_SAMPLES = 5000;

    public static CategoricalInfo generateFullInfo() {
        CategoricalInfo newInfo = new CategoricalInfo("test");

        for (int i = 0; i < 5; i++) {
            newInfo.observe(-1.0);
            newInfo.observe(2.0);
            newInfo.observe(3.0);
            newInfo.observe(4.0);
        }

        return newInfo;
    }

    /**
     * Generates an emtpy info which hasn't observed any values.
     * @return An empty info
     */
    public static CategoricalInfo generateEmptyInfo() {
        return new CategoricalInfo("empty");
    }

    public static CategoricalInfo generateOneValueInfo() {
        CategoricalInfo newInfo = new CategoricalInfo("one-value");

        for (int i = 0; i < 25; i++) {
            newInfo.observe(5);
        }

        return newInfo;
    }

    public void checkValueAndProb(CategoricalInfo c, double value, double probability) {
        int idx = -1;
        for (int i = 0; i < c.values.length; i++) {
            if (Math.abs(c.values[i] - value) < DELTA) {
                if (idx > 0) {
                    fail("Found value " + value + " at " + idx + " and " + i);
                } else {
                    idx = i;
                }
            }
        }
        double testProb = idx == 0 ? c.cdf[0] : c.cdf[idx] - c.cdf[idx-1];
        assertEquals(probability, testProb, DELTA);
    }

    @Test
    public void samplingTest() {
        SplittableRandom rng = new SplittableRandom(1);

        CategoricalInfo c;

        c = generateEmptyInfo();

        c.frequencyBasedSample(rng, 50);

        assertEquals(1,c.values.length);
        assertEquals(0,c.values[0],DELTA);
        assertEquals(1,c.cdf.length);
        assertEquals(1.0,c.cdf[0],DELTA);

        for (int i = 0; i < 50; i++) {
            assertEquals(0.0,c.frequencyBasedSample(rng, 50),DELTA);
        }

        c = generateOneValueInfo();

        c.frequencyBasedSample(rng, 50);

        assertEquals(2,c.values.length);
        assertEquals(0,c.values[0],DELTA);
        assertEquals(5,c.values[1],DELTA);
        assertEquals(2,c.values.length);
        assertEquals(0.5,c.cdf[0],DELTA);
        assertEquals(1.0,c.cdf[1],DELTA);

        double sum = 0;
        double posCount = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            double sample = c.frequencyBasedSample(rng, 50);
            if (sample > DELTA) {
                posCount++;
            }
            sum += sample;
        }
        assertEquals(0.5,posCount/NUM_SAMPLES,1e-1);
        assertEquals(2.5,sum/NUM_SAMPLES,1e-1);

        c = generateOneValueInfo();

        c.frequencyBasedSample(rng,1000);

        assertEquals(2,c.values.length);
        assertEquals(0,c.values[0],DELTA);
        assertEquals(5,c.values[1],DELTA);
        assertEquals(2,c.values.length);
        assertEquals(0.975,c.cdf[0],DELTA);
        assertEquals(1.0,c.cdf[1],DELTA);

        c = generateFullInfo();

        c.frequencyBasedSample(rng,50);

        assertEquals(5,c.values.length);
        assertEquals(5,c.cdf.length);
        checkValueAndProb(c,0.0,0.6);
        checkValueAndProb(c,-1.0,0.1);
        checkValueAndProb(c,2.0,0.1);
        checkValueAndProb(c,3.0,0.1);
        checkValueAndProb(c,4.0,0.1);

        c = generateFullInfo();

        c.frequencyBasedSample(rng, 100);

        assertEquals(5,c.values.length);
        assertEquals(5,c.cdf.length);
        checkValueAndProb(c,0.0,0.8);
        checkValueAndProb(c,-1.0,0.05);
        checkValueAndProb(c,2.0,0.05);
        checkValueAndProb(c,3.0,0.05);
        checkValueAndProb(c,4.0,0.05);
    }

    @Test
    public void equalityTest() {
        CategoricalInfo fullFirst = generateFullInfo();
        CategoricalInfo fullSecond = generateFullInfo();
        CategoricalInfo emptyFirst = generateEmptyInfo();
        CategoricalInfo emptySecond = generateEmptyInfo();
        CategoricalInfo oneFirst = generateOneValueInfo();
        CategoricalInfo oneSecond = generateOneValueInfo();

        assertEquals(fullFirst,fullSecond);
        assertEquals(emptyFirst,emptySecond);
        assertEquals(oneFirst,oneSecond);

        assertNotEquals(fullFirst,emptyFirst);
        assertNotEquals(fullFirst,oneFirst);
    }

}
