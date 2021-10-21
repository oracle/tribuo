/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.tests.csv;

import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.MultiLabelFactory;
import org.tribuo.regression.RegressionFactory;
import org.tribuo.regression.Regressor;
import org.tribuo.tests.Resources;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CSVLoaderWithMultiOutputsTest {

    @Test
    public void loadsMultiLabel() throws IOException {
        Path path = Resources.copyResourceToTmp("/org/tribuo/tests/csv/multilabel.csv");
        Set<String> responses = new HashSet<>(Arrays.asList("R1", "R2"));
        CSVLoader<MultiLabel> loader = new CSVLoader<>(new MultiLabelFactory());
        MutableDataset<MultiLabel> data = loader.load(path, responses);
        assertEquals(5, data.size());

        Example<MultiLabel> example = data.getExample(0);
        MultiLabel y = example.getOutput();
        assertTrue(y.contains("R1"));
        assertFalse(y.contains("R2"));
        assertEquals(1.0, example.lookup("A").getValue(), 1e-7);

        //
        // Row #1: R1=True, R2=True
        assertTrue(data.getExample(1).getOutput().contains("R1"));
        assertTrue(data.getExample(1).getOutput().contains("R2"));

        //
        // Row #2: R1=False and R2=False.
        // In this case, the labelSet is empty and the labelString is the empty string.
        assertEquals(0, data.getExample(2).getOutput().getLabelSet().size());
        assertEquals("", data.getExample(2).getOutput().getLabelString());
        assertTrue(data.getExample(2).validateExample());
    }

    @Test
    public void loadsMultiRegressor() throws IOException {
        Path path = Resources.copyResourceToTmp("/org/tribuo/tests/csv/multioutput-regression.csv");
        CSVLoader<Regressor> loader = new CSVLoader<>(new RegressionFactory());
        String[] responseNames = new String[]{"R1", "R2"};
        MutableDataset<Regressor> data = loader.load(path, new HashSet<>(Arrays.asList(responseNames)));
        assertEquals(5, data.size());

        Example<Regressor> x0 = data.getExample(0);
        assertArrayEquals(responseNames, x0.getOutput().getNames());
        assertArrayEquals(new double[]{0.1,0.2}, x0.getOutput().getValues());

        Example<Regressor> x1 = data.getExample(1);
        assertArrayEquals(responseNames, x1.getOutput().getNames());
        assertArrayEquals(new double[]{0.0,0.0}, x1.getOutput().getValues());
    }

    @Test
    public void loadsMultiRegressorWithoutHeader() throws IOException {
        Path path = Resources.copyResourceToTmp("/org/tribuo/tests/csv/multioutput-regression-noheader.csv");
        CSVLoader<Regressor> loader = new CSVLoader<>(new RegressionFactory());
        String[] header = {"A","B","C","D","R1","R2"};
        String[] responseNames = new String[]{"R1", "R2"};
        MutableDataset<Regressor> data = loader.load(path, new HashSet<>(Arrays.asList(responseNames)), header);
        assertEquals(5, data.size());

        Example<Regressor> x0 = data.getExample(0);
        assertArrayEquals(responseNames, x0.getOutput().getNames());
        assertArrayEquals(new double[]{0.1,0.2}, x0.getOutput().getValues());

        Example<Regressor> x1 = data.getExample(1);
        assertArrayEquals(responseNames, x1.getOutput().getNames());
        assertArrayEquals(new double[]{0.0,0.0}, x1.getOutput().getValues());
    }

}