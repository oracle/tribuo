/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.datasource;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.StreamSupport;

public class AggregateDataSourceTest {

    @Test
    public void testIterationOrder() {
        MockOutputFactory factory = new MockOutputFactory();
        String[] featureNames = new String[] {"X1","X2"};
        double[] featureValues = new double[] {1.0, 2.0};

        List<Example<MockOutput>> first = new ArrayList<>();
        first.add(new ArrayExample<>(new MockOutput("A"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("B"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("C"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("D"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("E"),featureNames,featureValues));
        ListDataSource<MockOutput> firstSource = new ListDataSource<>(first,factory,new SimpleDataSourceProvenance("First",factory));

        List<Example<MockOutput>> second = new ArrayList<>();
        second.add(new ArrayExample<>(new MockOutput("F"),featureNames,featureValues));
        second.add(new ArrayExample<>(new MockOutput("G"),featureNames,featureValues));
        ListDataSource<MockOutput> secondSource = new ListDataSource<>(second,factory,new SimpleDataSourceProvenance("Second",factory));

        List<Example<MockOutput>> third = new ArrayList<>();
        third.add(new ArrayExample<>(new MockOutput("H"),featureNames,featureValues));
        third.add(new ArrayExample<>(new MockOutput("I"),featureNames,featureValues));
        third.add(new ArrayExample<>(new MockOutput("J"),featureNames,featureValues));
        third.add(new ArrayExample<>(new MockOutput("K"),featureNames,featureValues));
        ListDataSource<MockOutput> thirdSource = new ListDataSource<>(third,factory,new SimpleDataSourceProvenance("Third",factory));

        List<DataSource<MockOutput>> sources = new ArrayList<>();
        sources.add(firstSource);
        sources.add(secondSource);
        sources.add(thirdSource);

        AggregateDataSource<MockOutput> adsSeq = new AggregateDataSource<>(sources, AggregateDataSource.IterationOrder.SEQUENTIAL);
        String[] expectedSeq = new String[] {"A","B","C","D","E","F","G","H","I","J","K"};
        String[] actualSeq = StreamSupport.stream(adsSeq.spliterator(), false).map(Example::getOutput).map(MockOutput::toString).toArray(String[]::new);
        Assertions.assertArrayEquals(expectedSeq,actualSeq);

        AggregateDataSource<MockOutput> adsRR = new AggregateDataSource<>(sources, AggregateDataSource.IterationOrder.ROUNDROBIN);
        String[] expectedRR = new String[] {"A","F","H","B","G","I","C","J","D","K","E"};
        String[] actualRR = StreamSupport.stream(adsRR.spliterator(), false).map(Example::getOutput).map(MockOutput::toString).toArray(String[]::new);
        Assertions.assertArrayEquals(expectedRR,actualRR);
    }

}
