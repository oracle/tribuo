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

package org.tribuo.evaluation;

import org.tribuo.DataSource;
import org.tribuo.test.MockDataSource;
import org.tribuo.test.MockOutput;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TrainTestSplitterTest {

    private static final long seed = 0L;

    @Test
    public void testSplitter_emptyDataSource() {
        DataSource<MockOutput> dataSource = new MockDataSource(0);

        TrainTestSplitter<MockOutput> splitter = new TrainTestSplitter<>(dataSource, 0.7, seed);
        assertEquals(0, splitter.totalSize());
        assertEquals(0, sizeOf(splitter.getTrain()));
        assertEquals(0, sizeOf(splitter.getTest()));
    }

    @Test
    public void testSplitter_singletonDataSource() {
        DataSource<MockOutput> dataSource = new MockDataSource(1);
        TrainTestSplitter<MockOutput> splitter = new TrainTestSplitter<>(dataSource, 0.7, seed);
        assertEquals(1, splitter.totalSize());
        assertEquals(0, sizeOf(splitter.getTrain()));
        assertEquals(1, sizeOf(splitter.getTest()));
    }

    @Test
    public void testSplitter() {
        DataSource<MockOutput> dataSource = new MockDataSource(10);
        TrainTestSplitter<MockOutput> splitter = new TrainTestSplitter<>(dataSource, 0.5, seed);
        assertEquals(10, splitter.totalSize());
        assertEquals(5, sizeOf(splitter.getTrain()));
        assertEquals(5, sizeOf(splitter.getTest()));
    }

    @Test
    public void testSplitter_indivisibleTrainProportion() {
        DataSource<MockOutput> dataSource = new MockDataSource(11);
        TrainTestSplitter<MockOutput> splitter = new TrainTestSplitter<>(dataSource, 0.5, seed);
        assertEquals(11, splitter.totalSize());
        assertEquals(5, sizeOf(splitter.getTrain()));
        assertEquals(6, sizeOf(splitter.getTest()));
    }

    private static <T> List<T> list(Iterable<T> iter) {
        List<T> l = new ArrayList<>();
        iter.forEach(l::add);
        return l;
    }

    private static <T> int sizeOf(Iterable<T> iter) {
        int ct = 0;
        for (T i : iter) { ct++; }
        return ct;
    }

}