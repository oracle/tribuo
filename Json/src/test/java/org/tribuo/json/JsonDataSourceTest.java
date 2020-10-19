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

package org.tribuo.json;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance;
import org.junit.jupiter.api.Test;
import org.tribuo.MutableDataset;
import org.tribuo.data.columnar.FieldExtractor;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.extractors.DateExtractor;
import org.tribuo.data.columnar.extractors.IntExtractor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.field.TextFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class JsonDataSourceTest {

    private static RowProcessor<MockOutput> buildRowProcessor() {
        Map<String, FieldProcessor> fieldProcessors = new HashMap<>();
        fieldProcessors.put("height",new DoubleFieldProcessor("height"));
        fieldProcessors.put("description",new TextFieldProcessor("description",new BasicPipeline(new BreakIteratorTokenizer(Locale.US),2)));
        fieldProcessors.put("transport",new IdentityProcessor("transport"));

        Map<String,FieldProcessor> regexMappingProcessors = new HashMap<>();
        regexMappingProcessors.put("extra.*", new DoubleFieldProcessor("regex"));

        ResponseProcessor<MockOutput> responseProcessor = new FieldResponseProcessor<>("disposition","UNK",new MockOutputFactory());

        List<FieldExtractor<?>> metadataExtractors = new ArrayList<>();
        metadataExtractors.add(new IntExtractor("id"));
        metadataExtractors.add(new DateExtractor("timestamp","timestamp","dd/MM/yyyy HH:mm"));

        return new RowProcessor<>(metadataExtractors,null,responseProcessor,fieldProcessors,regexMappingProcessors, Collections.emptySet());
    }

    @Test
    public void loadTest() throws URISyntaxException {
        URI dataFile = JsonDataSourceTest.class.getResource("/org/tribuo/json/test.json").toURI();

        RowProcessor<MockOutput> rowProcessor = buildRowProcessor();

        JsonDataSource<MockOutput> source = new JsonDataSource<>(dataFile, rowProcessor, true);

        MutableDataset<MockOutput> dataset = new MutableDataset<>(source);

        assertEquals(20,dataset.size(),"Found an incorrect number of rows when loading the json file.");

        DatasetProvenance prov = dataset.getProvenance();

        List<ObjectMarshalledProvenance> datasetProvenance = ProvenanceUtil.marshalProvenance(prov);

        assertFalse(datasetProvenance.isEmpty());

        ObjectProvenance unmarshalledProvenance = ProvenanceUtil.unmarshalProvenance(datasetProvenance);

        assertEquals(prov,unmarshalledProvenance);
    }

    @Test
    public void loadEmptyTest() throws URISyntaxException {
        URI emptyFile = JsonDataSourceTest.class.getResource("/org/tribuo/json/empty.json").toURI();

        RowProcessor<MockOutput> rowProcessor = buildRowProcessor();

        JsonDataSource<MockOutput> source = new JsonDataSource<>(emptyFile, rowProcessor, true);

        try {
            MutableDataset<MockOutput> dataset = new MutableDataset<>(source);
            fail("Empty files throw as they don't have any column headers");
        } catch (IllegalStateException e) {
            assertTrue(e.getMessage().contains("No elements found in JSON array"));
        }
    }

}
