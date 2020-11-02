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

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.OutputFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 *
 */
public class CSVDataSourceTest {

    private URI dataFile;
    private String[] headers;
    private RowProcessor<MockOutput> rowProcessor;

    private URI regexDataFile;
    private RowProcessor<MockOutput> regexRowProcessor;

    @BeforeEach
    public void setUp() throws URISyntaxException {
        dataFile = CSVDataSourceTest.class.getResource("/org/tribuo/data/csv/test.csv").toURI();
        headers = new String[] {"A", "B", "C", "D"};

        OutputFactory<MockOutput> outputFactory = new MockOutputFactory();
        ResponseProcessor<MockOutput> responseProcessor = new FieldResponseProcessor<>("RESPONSE", "", outputFactory);
        Map<String, FieldProcessor> processors = new HashMap<>();
        for(String header : headers) {
            processors.put(header, new IdentityProcessor(header));
        }

        rowProcessor = new RowProcessor<>(responseProcessor,processors);

        regexDataFile = CSVDataSourceTest.class.getResource("/org/tribuo/data/csv/test-regexmapping.csv").toURI();
        Map<String, FieldProcessor> regexProcessor = new HashMap<>();
        regexProcessor.put("foo.*", new IdentityProcessor("REGEXDUMMY"));

        regexRowProcessor = new RowProcessor<>(Collections.emptyList(), null, responseProcessor, Collections.emptyMap(), regexProcessor, Collections.emptySet());
    }

    @Test
    public void testBasic() {
        CSVDataSource<MockOutput> dataSource = new CSVDataSource<>(dataFile, rowProcessor, true);

        MutableDataset<MockOutput> dataset = new MutableDataset<>(dataSource);

        assertEquals(6,dataset.size(),"Found an incorrect number of rows when loading the csv.");

        DatasetProvenance prov = dataset.getProvenance();

        List<ObjectMarshalledProvenance> datasetProvenance = ProvenanceUtil.marshalProvenance(prov);

        assertFalse(datasetProvenance.isEmpty());

        ObjectProvenance unmarshalledProvenance = ProvenanceUtil.unmarshalProvenance(datasetProvenance);

        assertEquals(prov,unmarshalledProvenance);
    }

    @Test
    public void testBOM() throws URISyntaxException {
        // Excel inserts a BOM into UTF-8 formatted CSV files, this test checks that Tribuo correctly ignores it.
        URI bomDataFile = CSVDataSourceTest.class.getResource("/org/tribuo/data/csv/test-bom.csv").toURI();

        // Load the file with the BOM
        CSVDataSource<MockOutput> bomSource = new CSVDataSource<>(bomDataFile, rowProcessor, true);
        MutableDataset<MockOutput> bomDataset = new MutableDataset<>(bomSource);

        // Check the rows loaded correctly
        assertEquals(6,bomDataset.size(),"Found an incorrect number of rows when loading the csv.");

        // Check the feature space is the same
        assertEquals(13,bomDataset.getFeatureMap().size());

        // Load the clean file
        CSVDataSource<MockOutput> dataSource = new CSVDataSource<>(dataFile, rowProcessor, true);
        MutableDataset<MockOutput> dataset = new MutableDataset<>(dataSource);

        // Check the data is the same
        for (int i = 0; i < 6; i++) {
            assertEquals(dataset.getExample(i), bomDataset.getExample(i));
        }
    }

    @Test
    public void testRegexExpand() {
        CSVDataSource<MockOutput> dataSource = new CSVDataSource<>(regexDataFile, regexRowProcessor, true);

        MutableDataset<MockOutput> dataset = new MutableDataset<>(dataSource);

        assertEquals(6,dataset.size(),"Found an incorrect number of rows when loading the csv.");

        DatasetProvenance prov = dataset.getProvenance();

        List<ObjectMarshalledProvenance> datasetProvenance = ProvenanceUtil.marshalProvenance(prov);

        assertFalse(datasetProvenance.isEmpty());

        ObjectProvenance unmarshalledProvenance = ProvenanceUtil.unmarshalProvenance(datasetProvenance);

        assertEquals(prov,unmarshalledProvenance);

        assertEquals(13, dataset.getFeatureMap().size());
    }

    @Test
    public void testRowProcessorReuse() {
        Dataset<MockOutput> ds1 = new MutableDataset<>(new CSVDataSource<>(regexDataFile, regexRowProcessor, true));
        Dataset<MockOutput> ds2 = new MutableDataset<>(new CSVDataSource<>(regexDataFile, regexRowProcessor, true));
    }

}
