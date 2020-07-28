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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.tribuo.test.Helpers.mkExample;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.dataset.MinimumCardinalityDataset;
import org.tribuo.impl.ListExample;
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

        Dataset<MockOutput> prunedDataset = new MinimumCardinalityDataset<>(dataset, 2);
        infoMap = prunedDataset.getFeatureIDMap();
        assertEquals(4, infoMap.get("f1").getCount());
        assertNull(infoMap.get("f2"));
        assertEquals(2, infoMap.get("f3").getCount());
        assertNull(infoMap.get("f4"));
        assertNull(infoMap.get("f5"));
        
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

    }
}