/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.common.tree.impl;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

public class IntArrayContainerTest {

    @Test
    public void mergeListTest() {
        IntArrayContainer first = new IntArrayContainer(10);
        IntArrayContainer second = new IntArrayContainer(10);
        List<int[]> list = new ArrayList<>();
        list.add(new int[]{5, 7, 9});
        list.add(new int[]{1, 2, 4, 6});
        list.add(new int[]{3, 8, 10});

        int[] output = IntArrayContainer.merge(list, first, second);

        int[] expectedOutput = new int[]{1,2,3,4,5,6,7,8,9,10};

        Assertions.assertArrayEquals(expectedOutput, output);
    }

    @Test
    public void mergeTest() {
        IntArrayContainer first = new IntArrayContainer(10);
        IntArrayContainer second = new IntArrayContainer(10);
        first.fill(new int[]{1,2,3,5,7,9});
        int[] other = new int[]{0,4,6,8,10};

        IntArrayContainer.merge(first, other, second);

        int[] expectedOutput = new int[]{0,1,2,3,4,5,6,7,8,9,10};

        int[] output = second.copy();

        Assertions.assertArrayEquals(expectedOutput, output);
    }

    @Test
    public void fillTest() {
        IntArrayContainer first = new IntArrayContainer(5);
        IntArrayContainer second = new IntArrayContainer(10);
        int[] expected = new int[]{1,2,3,5,7,9};
        first.fill(expected);
        Assertions.assertArrayEquals(expected, first.copy());

        second.fill(first);
        Assertions.assertArrayEquals(expected, second.copy());

        IntArrayContainer third = new IntArrayContainer(3);
        third.fill(first);
        Assertions.assertArrayEquals(expected, third.copy());
    }

    @Test
    public void growTest() {
        IntArrayContainer first = new IntArrayContainer(10);
        int[] expected = new int[]{1,2,3,5,7,9};
        first.fill(expected);
        Assertions.assertArrayEquals(expected, first.copy());

        first.grow(20);
        Assertions.assertEquals(20,first.array.length);
        Assertions.assertArrayEquals(expected, first.copy());
    }

    @Test
    public void removeOtherTest() {
        IntArrayContainer first = new IntArrayContainer(10);
        IntArrayContainer second = new IntArrayContainer(10);
        first.fill(new int[]{1,2,3,5,7,9,10});

        int[] removed = new int[]{2,6,8,9};

        IntArrayContainer.removeOther(first, removed, second);

        int[] expected = new int[]{1,3,5,7,10};

        Assertions.assertArrayEquals(expected, second.copy());
    }

}
