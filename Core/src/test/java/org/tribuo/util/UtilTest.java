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

package org.tribuo.util;


import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
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
    public void testArgmin() {
        assertThrows(IllegalArgumentException.class, () -> Util.argmin(new ArrayList<Double>()));

        List<Integer> lst = Collections.singletonList(1);
        Pair<Integer, Integer> argmin = Util.argmin(lst);
        assertEquals(0, argmin.getA());
        assertEquals(1, argmin.getB());

        lst = Arrays.asList(3, 2, 1);
        argmin = Util.argmin(lst);
        assertEquals(2, argmin.getA());
        assertEquals(1, argmin.getB());
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
    public void testBinarySearch(){

        List<String> stringList = Arrays.asList("apple", "banana", "cherry", "date", "fig", "grape");
        List<Integer> intList = List.of(1, 3, 5, 7, 9);

        assertEquals( -1, Util.binarySearch(stringList, "Cherry"));
        assertEquals( 2, Util.binarySearch(stringList, "cherry"));

        assertEquals(2, Util.binarySearch(intList, 5, x -> x));
        assertEquals(-4, Util.binarySearch(intList, 6, x -> x));

    }

    @Test
    public void testSortedDifference() {
        assertTimeoutPreemptively(Duration.ofSeconds(10), () -> {
            assertArrayEquals(new int[]{1, 3}, Util.sortedDifference(new int[]{1, 2, 3}, new int[]{2}));
            assertArrayEquals(new int[]{1, 3}, Util.sortedDifference(new int[]{1, 3}, new int[]{2}));
            assertArrayEquals(new int[]{1}, Util.sortedDifference(new int[]{1}, new int[]{5}));
            assertArrayEquals(new int[]{5}, Util.sortedDifference(new int[]{5}, new int[]{1}));
            assertArrayEquals(new int[]{}, Util.sortedDifference(new int[]{1, 2}, new int[]{1, 2}));
            assertArrayEquals(new int[]{1, 2}, Util.sortedDifference(new int[]{1, 2}, new int[]{}));
            assertArrayEquals(new int[]{}, Util.sortedDifference(new int[]{}, new int[]{1, 2}));
            assertArrayEquals(new int[]{0, 4, 7}, Util.sortedDifference(new int[]{0, 2, 4, 6, 7}, new int[]{1, 2, 3, 6, 8}));

            // Compare against a simple filter for all strictly increasing arrays with values in 0..4.
            List<int[]> arrays = new ArrayList<>();
            for (int mask = 0; mask < 32; mask++) {
                int[] arr = new int[Integer.bitCount(mask)];
                int k = 0;
                for (int v = 0; v < 5; v++) {
                    if ((mask & (1 << v)) != 0) {
                        arr[k++] = v;
                    }
                }
                arrays.add(arr);
            }
            for (int[] first : arrays) {
                for (int[] second : arrays) {
                    int[] expected = Arrays.stream(first).filter(x -> Arrays.binarySearch(second, x) < 0).toArray();
                    assertArrayEquals(expected, Util.sortedDifference(first, second),
                            () -> Arrays.toString(first) + " minus " + Arrays.toString(second));
                }
            }
        });
    }

}
