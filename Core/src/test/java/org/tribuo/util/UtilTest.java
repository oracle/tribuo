/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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


import com.oracle.labs.mlrg.olcut.util.MutableLong;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class UtilTest {
    private static final double DELTA = 1e-12;

    @Test
    public void testArgmax() {
        assertThrows(IllegalArgumentException.class, () -> Util.argmax(new ArrayList<Double>()));

        List<Integer> lst = Collections.singletonList(1);
        Pair<Integer, Integer> argmax = Util.argmax(lst);
        assertEquals(0, argmax.getA());
        assertEquals(1, argmax.getB());

        lst = Arrays.asList(3, 2, 1);
        argmax = Util.argmax(lst);
        assertEquals(0, argmax.getA());
        assertEquals(3, argmax.getB());
    }

    @Test
    public void testAUC() {
        double output;

        try {
            output = Util.auc(new double[]{0.0,1.0},new double[]{0.0,1.0,2.0});
            fail("Exception not thrown for mismatched lengths.");
        } catch (IllegalArgumentException e) { }

        try {
            output = Util.auc(new double[]{0.0,1.0,2.0,1.5,3.0}, new double[]{1.0,1.0,1.0,1.0,1.0});
            fail("Exception not thrown for non-increasing x.");
        } catch (IllegalStateException e) { }

        output = Util.auc(new double[]{4,6,8},new double[]{1,2,3});
        assertEquals(8.0,output,DELTA);

        output = Util.auc(new double[]{0,1},new double[]{0,1});
        assertEquals(0.5,output,DELTA);

        output = Util.auc(new double[]{0,0,1},new double[]{1,1,0});
        assertEquals(0.5,output,DELTA);

        output = Util.auc(new double[]{0,1},new double[]{1,1});
        assertEquals(1,output,DELTA);

        output = Util.auc(new double[]{0,0.5,1},new double[]{0,0.5,1});
        assertEquals(0.5,output,DELTA);
    }

    @Test
    public void testSampleFromCDF() {
        double[] pmf = new double[]{0.1,0.2,0.0,0.3,0.0,0.0,0.4,0.0};
        double[] cdf = Util.generateCDF(pmf);

        double[] expectedCDF = new double[]{0.1,0.3,0.3,0.6,0.6,0.6,1.0,1.0};

        assertArrayEquals(expectedCDF,cdf,1e-10);

        SplittableRandom rng = new SplittableRandom(1235L);

        Map<Integer, MutableLong> counter = new HashMap<>();

        final int numSamples = 10000;
        for (int i = 0; i < numSamples; i++) {
            int curSample = Util.sampleFromCDF(cdf,rng);
            MutableLong l = counter.computeIfAbsent(curSample, k -> new MutableLong());
            l.increment();
        }

        assertNotNull(counter.get(0));
        assertNotNull(counter.get(1));
        assertNull(counter.get(2));
        assertNotNull(counter.get(3));
        assertNull(counter.get(4));
        assertNull(counter.get(5));
        assertNotNull(counter.get(6));
        assertNull(counter.get(7));

        double total = 0;
        for (Map.Entry<Integer, MutableLong> e : counter.entrySet()) {
            total += e.getValue().longValue();
        }
        assertEquals(numSamples,total);
        assertEquals(counter.get(0).longValue()/total,0.1,1e-1);
        assertEquals(counter.get(1).longValue()/total,0.2,1e-1);
        assertEquals(counter.get(3).longValue()/total,0.3,1e-1);
        assertEquals(counter.get(6).longValue()/total,0.4,1e-1);
    }
}
