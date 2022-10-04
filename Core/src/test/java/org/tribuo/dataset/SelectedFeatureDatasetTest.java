/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.dataset;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.SelectedFeatureSet;
import org.tribuo.impl.ArrayExample;
import org.tribuo.test.Helpers;
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockFeatureSelector;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SelectedFeatureDatasetTest {

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(SelectedFeatureDataset.class.getName());
        logger.setLevel(Level.WARNING);
    }

    public static Dataset<MockOutput> createDataset() {
        MockOutputFactory outputFactory = new MockOutputFactory();

        MutableDataset<MockOutput> dataset = new MutableDataset<>(new MockDataSourceProvenance(),outputFactory);

        String A = "A", B = "B", C = "C", D = "D", E = "E";
        MockOutput o = new MockOutput("tmp");
        Example<MockOutput> ex = new ArrayExample<>(o,new String[]{A,B,C,D,E},new double[]{1,1,1,1,1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{A,C,E},new double[]{1,1,1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{C,D},new double[]{1,1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{B,C,E},new double[]{1,1,1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{A,B},new double[]{1,1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{A},new double[]{1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{B},new double[]{1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{C},new double[]{1});
        dataset.add(ex);
        ex = new ArrayExample<>(o,new String[]{D},new double[]{1});
        dataset.add(ex);

        return dataset;
    }

    @Test
    public void testSelection() {
        Dataset<MockOutput> data = createDataset();

        MockFeatureSelector f = new MockFeatureSelector(Arrays.asList("A","B","E"));

        SelectedFeatureSet sfs = f.select(data);

        Helpers.testProtoSerialization(sfs);

        SelectedFeatureDataset<MockOutput> selected = new SelectedFeatureDataset<>(data,sfs);

        assertEquals(3, selected.getNumExamplesRemoved());
        assertEquals(6, selected.size());

        ImmutableFeatureMap ifm = selected.getFeatureIDMap();
        assertEquals(3,ifm.size());
        assertEquals(4,ifm.get("A").getCount());
        assertEquals(4,ifm.get("B").getCount());
        assertEquals(3,ifm.get("E").getCount());

        Helpers.testProvenanceMarshalling(selected.getProvenance());

        SelectedFeatureDataset<MockOutput> deser = (SelectedFeatureDataset<MockOutput>) Helpers.testDatasetSerialization(selected);
        assertEquals(sfs, deser.getFeatureSet());
    }

}
