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

package org.tribuo.data.csv;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.junit.jupiter.api.Test;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.test.MockMultiOutput;
import org.tribuo.test.MockMultiOutputFactory;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;


public class CSVLoaderTest {

    @Test
    public void testLoad() throws IOException {
        URL path = CSVLoaderTest.class.getResource("/org/tribuo/data/csv/test.csv");
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        checkDataTestCsv(loader.loadDataSource(path, "RESPONSE"));
        checkDataTestCsv(loader.loadDataSource(path, Collections.singleton("RESPONSE")));
    }

    @Test
    public void testLoadBOM() throws IOException {
        URL path = CSVLoaderTest.class.getResource("/org/tribuo/data/csv/test-bom.csv");
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        checkDataTestCsv(loader.loadDataSource(path, "RESPONSE"));
        checkDataTestCsv(loader.loadDataSource(path, Collections.singleton("RESPONSE")));
    }

    @Test
    public void testLoadMultiOutput() throws IOException {
        URL path = CSVLoaderTest.class.getResource("/org/tribuo/data/csv/test-multioutput.csv");
        Set<String> responses = new HashSet<>(Arrays.asList("R1", "R2"));
        CSVLoader<MockMultiOutput> loader = new CSVLoader<>(new MockMultiOutputFactory());
        DataSource<MockMultiOutput> source = loader.loadDataSource(path, responses);
        MutableDataset<MockMultiOutput> data = new MutableDataset<>(source);
        assertEquals(6, data.size());

        Example<MockMultiOutput> example = data.getExample(0);
        MockMultiOutput y = example.getOutput();
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
    public void testLoadNoHeader() throws IOException {
        URL path = CSVLoader.class.getResource("/org/tribuo/data/csv/test.csv");
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        //
        // Currently, passing a the header into loader.load when the CSV has a header row will cause an error. This is
        // because CSVIterator does not skip the first line in this case.
        // TODO do we want this behavior?
        String[] header = new String[]{"A","B","C","D","RESPONSE"};
        assertThrows(NumberFormatException.class, () -> loader.load(Paths.get(path.toURI()), "RESPONSE", header));
        assertThrows(NumberFormatException.class, () -> loader.load(Paths.get(path.toURI()), Collections.singleton("RESPONSE"), header));
        //
        // Test behavior when CSV file does not have a header row and the user instead supplies the header.
        URL noheader = CSVLoader.class.getResource("/org/tribuo/data/csv/test-noheader.csv");
        DataSource<MockOutput> source = loader.loadDataSource(noheader, "RESPONSE", header);

        // Check that the source persisted the headers in the provenance
        CSVDataSource.CSVDataSourceProvenance prov = (CSVDataSource.CSVDataSourceProvenance) source.getProvenance();
        Provenance headerProv = prov.getConfiguredParameters().get("headers");
        assertTrue(headerProv instanceof ListProvenance);
        @SuppressWarnings("unchecked")
        ListProvenance<StringProvenance> listProv = (ListProvenance<StringProvenance>) headerProv;
        assertEquals(header.length,listProv.getList().size());
        assertEquals(Arrays.asList(header),listProv.getList().stream().map(StringProvenance::getValue).collect(Collectors.toList()));

        // Check the data loaded correctly.
        checkDataTestCsv(source);
        checkDataTestCsv(loader.loadDataSource(noheader, Collections.singleton("RESPONSE"), header));
    }

    private static void checkDataTestCsv(DataSource<MockOutput> source) {
        MutableDataset<MockOutput> data = new MutableDataset<>(source);
        assertEquals(6, data.size());
        assertEquals("monkey", data.getExample(0).getOutput().label);
        assertEquals(1.0, data.getExample(0).lookup("A").getValue(), 1e-7);
        for (Example<MockOutput> x : data.getData()) {
            assertEquals(4, x.size());
        }
    }

    @Test
    public void testLoadBadHeader() {
        URL path = CSVLoader.class.getResource("/org/tribuo/data/csv/test-noheader.csv");
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        //
        // Missing feature column "A"
        assertThrows(IllegalArgumentException.class,
                () -> loader.load(Paths.get(path.toURI()), Collections.singleton("RESPONSE"), new String[]{"B","C","D","RESPONSE"}));
        //
        // Missing "RESPONSE" column
        assertThrows(IllegalArgumentException.class,
                () -> loader.loadDataSource(path, Collections.singleton("RESPONSE"), new String[]{"A","B","C","D"}));
    }

    /**
     * Tests behavior when an invalid response name is passed.
     */
    @Test
    public void testInvalidResponseName() {
        URL path = CSVLoader.class.getResource("/org/tribuo/data/csv/test.csv");
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        assertThrows(IllegalArgumentException.class, () -> loader.loadDataSource(path, ""));
        assertThrows(IllegalArgumentException.class, () -> loader.loadDataSource(path, "response"));
        assertThrows(IllegalArgumentException.class, () -> loader.loadDataSource(path, "sdfas"));
        String tmp = null;
        assertThrows(IllegalArgumentException.class, () -> loader.loadDataSource(path, tmp));
        assertThrows(IllegalArgumentException.class, () -> loader.loadDataSource(path, Collections.singleton("alkjfd")));
        assertThrows(IllegalArgumentException.class, () -> loader.loadDataSource(path, new HashSet<>()));
    }

    /**
     * Tests behavior when loading a file that does not exist
     */
    @Test
    public void testFileDoesNotExist() {
        Path tmp;
        try {
            tmp = Files.createTempFile("CSVLoaderTest_testFileDoesNotExist", "txt");
            Files.delete(tmp);
            assertFalse(Files.exists(tmp));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        assertThrows(NoSuchFileException.class, () -> loader.load(tmp, "RESPONSE"));
    }
    
    /**
     * Tests behavior when a csv with a single output response column is loaded as though it were a multi-output problem.
     */
    @Test
    public void testLoadSingleOutputAsMultiOutput() throws IOException {
        URL path = CSVLoader.class.getResource("/org/tribuo/data/csv/test.csv");
        CSVLoader<MockMultiOutput> loader = new CSVLoader<>(new MockMultiOutputFactory());
        DataSource<MockMultiOutput> source = loader.loadDataSource(path, "RESPONSE");
        MutableDataset<MockMultiOutput> data = new MutableDataset<>(source);
        assertEquals(6, data.size());
        assertEquals("monkey", data.getExample(0).getOutput().getLabelString());
        assertEquals("monkey", data.getExample(1).getOutput().getLabelString());
        assertEquals("baboon", data.getExample(2).getOutput().getLabelString());
        for (Example<MockMultiOutput> x : data.getData()) {
            assertEquals(4, x.size());
        }
        //assertThrows(IllegalArgumentException.class, () -> loader.load(path, "RESPONSE"));
    }

    /**
     * Tests behavior when a csv with multiple output response columns is loaded as though it were a single-output problem.
     */
    @Test
    public void testLoadMultiOutputAsSingleOutput() throws IOException {
        URL path = CSVLoader.class.getResource("/org/tribuo/data/csv/test-multioutput.csv");
        Set<String> responses = new LinkedHashSet<>(Arrays.asList("R1", "R2"));
        CSVLoader<MockOutput> loader = new CSVLoader<>(new MockOutputFactory());
        DataSource<MockOutput> source = loader.loadDataSource(path, responses);
        MutableDataset<MockOutput> data = new MutableDataset<>(source);
        assertEquals(6, data.size());
        assertEquals("[R1=TRUE, R2=FALSE]", data.getExample(0).getOutput().label);
        assertEquals("[R1=TRUE, R2=TRUE]", data.getExample(1).getOutput().label);
        assertEquals("[R1=FALSE, R2=FALSE]", data.getExample(2).getOutput().label);
    }

}