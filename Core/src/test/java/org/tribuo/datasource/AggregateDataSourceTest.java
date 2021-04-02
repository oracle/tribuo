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
import org.tribuo.ConfigurableDataSource;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.SimpleDataSourceProvenance;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.StreamSupport;

public class AggregateDataSourceTest {

    @Test
    public void testADSIterationOrder() {
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
        Helpers.testProvenanceMarshalling(adsSeq.getProvenance());

        AggregateDataSource<MockOutput> adsRR = new AggregateDataSource<>(sources, AggregateDataSource.IterationOrder.ROUNDROBIN);
        String[] expectedRR = new String[] {"A","F","H","B","G","I","C","J","D","K","E"};
        String[] actualRR = StreamSupport.stream(adsRR.spliterator(), false).map(Example::getOutput).map(MockOutput::toString).toArray(String[]::new);
        Assertions.assertArrayEquals(expectedRR,actualRR);
        Helpers.testProvenanceMarshalling(adsRR.getProvenance());
    }

    @Test
    public void testACDSIterationOrder() {
        MockOutputFactory factory = new MockOutputFactory();
        String[] featureNames = new String[] {"X1","X2"};
        double[] featureValues = new double[] {1.0, 2.0};

        List<Example<MockOutput>> first = new ArrayList<>();
        first.add(new ArrayExample<>(new MockOutput("A"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("B"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("C"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("D"),featureNames,featureValues));
        first.add(new ArrayExample<>(new MockOutput("E"),featureNames,featureValues));
        MockListConfigurableDataSource<MockOutput> firstSource = new MockListConfigurableDataSource<>(first,factory,new SimpleDataSourceProvenance("First",factory));

        List<Example<MockOutput>> second = new ArrayList<>();
        second.add(new ArrayExample<>(new MockOutput("F"),featureNames,featureValues));
        second.add(new ArrayExample<>(new MockOutput("G"),featureNames,featureValues));
        MockListConfigurableDataSource<MockOutput> secondSource = new MockListConfigurableDataSource<>(second,factory,new SimpleDataSourceProvenance("Second",factory));

        List<Example<MockOutput>> third = new ArrayList<>();
        third.add(new ArrayExample<>(new MockOutput("H"),featureNames,featureValues));
        third.add(new ArrayExample<>(new MockOutput("I"),featureNames,featureValues));
        third.add(new ArrayExample<>(new MockOutput("J"),featureNames,featureValues));
        third.add(new ArrayExample<>(new MockOutput("K"),featureNames,featureValues));
        MockListConfigurableDataSource<MockOutput> thirdSource = new MockListConfigurableDataSource<>(third,factory,new SimpleDataSourceProvenance("Third",factory));

        List<ConfigurableDataSource<MockOutput>> sources = new ArrayList<>();
        sources.add(firstSource);
        sources.add(secondSource);
        sources.add(thirdSource);

        AggregateConfigurableDataSource<MockOutput> acdsSeq = new AggregateConfigurableDataSource<>(sources, AggregateDataSource.IterationOrder.SEQUENTIAL);
        String[] expectedSeq = new String[] {"A","B","C","D","E","F","G","H","I","J","K"};
        String[] actualSeq = StreamSupport.stream(acdsSeq.spliterator(), false).map(Example::getOutput).map(MockOutput::toString).toArray(String[]::new);
        Assertions.assertArrayEquals(expectedSeq,actualSeq);
        Helpers.testProvenanceMarshalling(acdsSeq.getProvenance());

        AggregateConfigurableDataSource<MockOutput> acdsRR = new AggregateConfigurableDataSource<>(sources, AggregateDataSource.IterationOrder.ROUNDROBIN);
        String[] expectedRR = new String[] {"A","F","H","B","G","I","C","J","D","K","E"};
        String[] actualRR = StreamSupport.stream(acdsRR.spliterator(), false).map(Example::getOutput).map(MockOutput::toString).toArray(String[]::new);
        Assertions.assertArrayEquals(expectedRR,actualRR);
        Helpers.testProvenanceMarshalling(acdsRR.getProvenance());

    }

    /**
     * This isn't actually configurable, it's used to test {@link AggregateConfigurableDataSource}.
     * @param <T> The output type.
     */
    private static class MockListConfigurableDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {

        private final List<Example<T>> data;

        private final OutputFactory<T> factory;

        private final DataSourceProvenance provenance;

        public MockListConfigurableDataSource(List<Example<T>> list, OutputFactory<T> factory, DataSourceProvenance provenance) {
            this.data = Collections.unmodifiableList(new ArrayList<>(list));
            this.factory = factory;
            this.provenance = provenance;
        }

        /**
         * Number of examples.
         * @return The number of examples.
         */
        public int size() {
            return data.size();
        }

        @Override
        public OutputFactory<T> getOutputFactory() {
            return factory;
        }

        @Override
        public DataSourceProvenance getProvenance() {
            return provenance;
        }

        @Override
        public Iterator<Example<T>> iterator() {
            return data.iterator();
        }

        @Override
        public String toString() {
            return provenance.toString();
        }
    }

}
