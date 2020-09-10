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

package org.tribuo.data.csv;

import org.tribuo.DataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.test.MockMultiOutput;
import org.tribuo.test.MockMultiOutputFactory;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class CSVSaverTest {

    @Test
    public void testSave() throws IOException {
        URL path = CSVSaverTest.class.getResource("/org/tribuo/data/csv/test.csv");
        Set<String> responses = Collections.singleton("RESPONSE");
        //
        // Load the csv
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        DataSource<MockOutput> source = loader.loadDataSource(path, responses);
        Dataset<MockOutput> before = new MutableDataset<>(source);
        //
        // Save the dataset
        CSVSaver saver = new CSVSaver();
        Path tmp = Files.createTempFile("foo","csv");
        tmp.toFile().deleteOnExit();
        saver.save(tmp, before, "RESPONSE");
        //
        // Reload and check that before & after are equivalent.
        MutableDataset<MockOutput> after = loader.load(tmp, responses);
        assertEquals(before.getData(), after.getData());
        assertEquals(before.getOutputIDInfo().size(), after.getOutputIDInfo().size());
        assertEquals(before.getFeatureIDMap().size(), after.getFeatureIDMap().size());
        for (int i = 0; i < before.size(); i++) {
            Example<MockOutput> b = before.getExample(i);
            Example<MockOutput> a = after.getExample(i);
            assertEquals(a,b);
        }
    }

    @Test
    public void testSaveMultiOutput() throws IOException {
        URL path = CSVSaverTest.class.getResource("/org/tribuo/data/csv/test-multioutput.csv");
        Set<String> responses = new HashSet<>(Arrays.asList("R1", "R2"));
        //
        // Load the csv
        CSVLoader<MockMultiOutput> loader = new CSVLoader<>(new MockMultiOutputFactory());
        DataSource<MockMultiOutput> source = loader.loadDataSource(path, responses);
        Dataset<MockMultiOutput> before = new MutableDataset<>(source);
        //
        // Save the dataset
        CSVSaver saver = new CSVSaver();
        Path tmp = Files.createTempFile("foo-multi","csv");
        tmp.toFile().deleteOnExit();
        saver.save(tmp, before, responses);
        //
        // Reload and check that before & after are equivalent.
        MutableDataset<MockMultiOutput> after = loader.load(tmp, responses);

        assertEquals(before.getData(), after.getData());
        assertEquals(before.getOutputIDInfo().size(), after.getOutputIDInfo().size());
        assertEquals(before.getFeatureIDMap().size(), after.getFeatureIDMap().size());
        for (int i = 0; i < before.size(); i++) {
            Example<MockMultiOutput> b = before.getExample(i);
            Example<MockMultiOutput> a = after.getExample(i);
            assertEquals(a,b);
        }
    }

    /**
     * Test behavior when dataset to save is empty.
     */
    @Test
    public void testSaveEmpty() throws IOException {
        MutableDataset<MockOutput> src = new MutableDataset<>(null, new MockOutputFactory());
        CSVSaver saver = new CSVSaver();
        Path tmp = Files.createTempFile("foo-empty","csv");
        tmp.toFile().deleteOnExit();
        saver.save(tmp, src, "RESPONSE");
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        MutableDataset<MockOutput> tgt = loader.load(tmp, "RESPONSE");
        assertEquals(0, tgt.size());
    }

}