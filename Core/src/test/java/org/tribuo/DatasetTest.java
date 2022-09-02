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

package org.tribuo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.tribuo.test.Helpers.mkExample;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.dataset.DatasetView;
import org.tribuo.dataset.MinimumCardinalityDataset;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.ListExample;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

/**
 **/
public class DatasetTest {

    @BeforeAll
    public static void suppressLogging() {
        Logger logger = Logger.getLogger(MinimumCardinalityDataset.class.getName());
        logger.setLevel(Level.WARNING);
    }

    /**
     * Tests that list returned from {@link Dataset#getData()} is unmodifiable.
     */
    @Test
    public void testGetData() {
        OutputFactory<MockOutput> outputFactory = new MockOutputFactory();
        MutableDataset<MockOutput> a = new MutableDataset<>(new MockDataSourceProvenance(), outputFactory);
        Assertions.assertThrows(UnsupportedOperationException.class, () -> a.getData().add(mkExample(outputFactory.generateOutput("1"), "a")), "Expected exception thrown as adding to unmodifiable list.");
    }
    
    @Test
    public void testMinimumCardinality() {
        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(), new MockOutputFactory());

        Example<MockOutput> ex1 = new ListExample<>(new MockOutput("green"));
        ex1.add(new Feature("f1", 1.0));
        ex1.add(new Feature("f2", 0.0));
        ex1.add(new Feature("f3", 1.0));
        dataset.add(ex1);
        Example<MockOutput> ex2 = new ListExample<>(new MockOutput("green"));
        ex2.add(new Feature("f1", 1.0));
        ex2.add(new Feature("f2", 0.0));
        ex2.add(new Feature("f3", 1.0));
        dataset.add(ex2);
        Example<MockOutput> ex3 = new ListExample<>(new MockOutput("blue"));
        ex3.add(new Feature("f1", 1.0));
        ex3.add(new Feature("f2", 0.0));
        ex3.add(new Feature("f4", 1.0));
        dataset.add(ex3);
        Example<MockOutput> ex4 = new ListExample<>(new MockOutput("green"));
        ex4.add(new Feature("f1", 1.0));
        ex4.add(new Feature("f4", 0.0));
        ex4.add(new Feature("f5", 1.0));
        dataset.add(ex4);

        FeatureMap infoMap = dataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertEquals(0, infoMap.get("f2").getCount());
        assertEquals(2, infoMap.get("f3").getCount());
        // One as non-sparse zeros are ignored
        assertEquals(1, infoMap.get("f4").getCount());
        assertEquals(1, infoMap.get("f5").getCount());

        MinimumCardinalityDataset<MockOutput> prunedDataset = new MinimumCardinalityDataset<>(dataset, 2);
        infoMap = prunedDataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertNull(infoMap.get("f2"));
        assertEquals(2, infoMap.get("f3").getCount());
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));
        MinimumCardinalityDataset<MockOutput> deser = (MinimumCardinalityDataset<MockOutput>) Helpers.testDatasetSerialization(prunedDataset);
        assertEquals(prunedDataset.getMinCardinality(), deser.getMinCardinality());
        
        ex2 = prunedDataset.getExample(1);
        Feature f1 = ex2.lookup("f1");
        assertEquals(1.0, f1.getValue(), 1e-5);
        assertNull(ex2.lookup("f5"));

        prunedDataset = new MinimumCardinalityDataset<>(dataset, 3);
        assertEquals(4, prunedDataset.size());
        infoMap = prunedDataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertNull(infoMap.get("f2"));
        assertNull(infoMap.get("f3"));
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));

        //no examples make it through the pruning
        prunedDataset = new MinimumCardinalityDataset<>(dataset, 5);
        assertEquals(0, prunedDataset.size());
        infoMap = prunedDataset.getFeatureIDMap();
        assertNull(infoMap.get("f1"));
        assertNull(infoMap.get("f2"));
        assertNull(infoMap.get("f3"));
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));
        Helpers.testDatasetSerialization(prunedDataset);
    }

    @Test
    public void testImmutable() {
        MockOutputFactory mockFactory = new MockOutputFactory();
        MockDataSourceProvenance mockProvenance = new MockDataSourceProvenance();
        MockOutput mockOutput = new MockOutput("test");

        List<Example<MockOutput>> examples = new ArrayList<>();
        examples.add(new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{1,1,1}));
        examples.add(new ArrayExample<>(mockOutput,new String[]{"a","b","c","d"},new double[]{1,1,1,1}));
        examples.add(new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{3,3,3}));
        examples.add(new ArrayExample<>(mockOutput,new String[]{"b","c"},new double[]{1,1}));

        MutableDataset<MockOutput> dataset = new MutableDataset<>(examples, mockProvenance, mockFactory);

        ImmutableDataset<MockOutput> imm = ImmutableDataset.copyDataset(dataset);

        Helpers.testDatasetSerialization(imm);

        DatasetView<MockOutput> view = DatasetView.createView(dataset, e -> {
            for (Feature f : e) { if (f.getName().equals("d")) { return true; } } return false;
        }, "d");

        DatasetView<MockOutput> deser = (DatasetView<MockOutput>) Helpers.testDatasetSerialization(view);
        assertEquals("d", deser.getTag());

        view = DatasetView.createBootstrapView(imm, 10, 12345);
        deser = (DatasetView<MockOutput>) Helpers.testDatasetSerialization(view);
    }

    @Test
    public void testDense() {
        MockOutputFactory mockFactory = new MockOutputFactory();
        MockDataSourceProvenance mockProvenance = new MockDataSourceProvenance();
        MockOutput mockOutput = new MockOutput("test");

        MutableDataset<MockOutput> dataset = new MutableDataset<>(mockProvenance, mockFactory);

        // Empty datasets are dense
        assertTrue(dataset.isDense());

        ArrayExample<MockOutput> first = new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{1,1,1});
        ArrayExample<MockOutput> second = new ArrayExample<>(mockOutput,new String[]{"a","b","c","d"},new double[]{1,1,1,1});
        ArrayExample<MockOutput> third = new ArrayExample<>(mockOutput,new String[]{"a","b","c"},new double[]{3,3,3});
        ArrayExample<MockOutput> fourth = new ArrayExample<>(mockOutput,new String[]{"b","c"},new double[]{1,1});

        dataset.add(first);
        MutableDataset<MockOutput> deser = (MutableDataset<MockOutput>) Helpers.testDatasetSerialization(dataset);

        // This example is dense
        assertTrue(dataset.isDense());
        assertTrue(deser.isDense());

        dataset.add(second);
        deser = (MutableDataset<MockOutput>) Helpers.testDatasetSerialization(dataset);

        // This example is dense, but it makes the previous one not dense as it adds a new feature
        assertFalse(dataset.isDense());
        assertFalse(deser.isDense());

        // flush out the previous test
        dataset.clear();

        dataset.add(first);
        dataset.add(third);

        // These examples are both dense
        assertTrue(dataset.isDense());

        // flush out old test
        dataset.clear();

        // Add all the examples, making it sparse
        dataset.add(first);
        dataset.add(second);
        dataset.add(third);
        dataset.add(fourth);

        // Should be sparse
        assertFalse(dataset.isDense());

        // Densify it
        dataset.densify();

        // Now it should be dense
        assertTrue(dataset.isDense());

        ArrayExample<MockOutput> fifth = new ArrayExample<>(mockOutput,new String[]{"a","b","c","d","e"},new double[]{1,1,1,1,1});

        // Makes the previous examples sparse
        dataset.add(fifth);
        assertFalse(dataset.isDense());

        dataset.densify();

        assertTrue(dataset.isDense());

        for (Example<MockOutput> e : dataset) {
            assertEquals(5,e.size());
        }

        Helpers.testDatasetSerialization(dataset);
    }
}