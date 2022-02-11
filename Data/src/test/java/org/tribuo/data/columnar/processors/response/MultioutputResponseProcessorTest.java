/*
 * Copyright (c) 2020-2022 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.data.columnar.processors.response;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.test.MockMultiOutput;
import org.tribuo.test.MockMultiOutputFactory;
import org.tribuo.test.MockOutputFactory;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MultioutputResponseProcessorTest {

    private URI dataFile;
    private URI missingDataFile;
    private Map<String, FieldProcessor> fieldProcessors;
    private List<Set<String>> multiTruthValues;


    private static <T extends Output<T>> RowProcessor<T> makeRowProcessor(ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessors) {
        return new RowProcessor<T>(responseProcessor, fieldProcessors);
    }

    private <T extends Output<T>, L> void doTest(ResponseProcessor<T> responseProcessor, List<L> expectedLabels, Function<T, L> labelMapper) {
        RowProcessor<T> rowProcessor = makeRowProcessor(responseProcessor, fieldProcessors);

        for (URI df: Arrays.asList(dataFile, missingDataFile)) {
            CSVDataSource<T> ds = new CSVDataSource<>(df, rowProcessor, true);

            Iterator<Example<T>> iter = ds.iterator();
            for(L l: expectedLabels) {
                assertEquals(l, labelMapper.apply(iter.next().getOutput()));
            }
        }
    }

    @BeforeEach
    public void setup() throws URISyntaxException {
        dataFile = MultioutputResponseProcessorTest.class.getResource("/org/tribuo/data/csv/test-multioutput.csv").toURI();
        missingDataFile = MultioutputResponseProcessorTest.class.getResource("/org/tribuo/data/csv/test-missingoutput.csv").toURI();
        fieldProcessors = new HashMap<>();
        for(String header: Arrays.asList("A", "B", "D")) {
            fieldProcessors.put(header, new IdentityProcessor(header));
        }
        multiTruthValues = Arrays.asList(
                new HashSet<>(Arrays.asList("R1")),
                new HashSet<>(Arrays.asList("R1", "R2")),
                new HashSet<>(Collections.emptyList()),
                new HashSet<>(Arrays.asList("R1")),
                new HashSet<>(Arrays.asList("R2")),
                new HashSet<>(Arrays.asList("R2")));
    }

    @Test
    public void binaryTest() {
        doTest(new BinaryResponseProcessor<>(
                        Arrays.asList("R1", "R2"),
                        Arrays.asList("TRUE", "TRUE"),
                        new MockMultiOutputFactory(),
                        "true", "false", true)
                , multiTruthValues
                , MockMultiOutput::getNameSet);

        doTest(new BinaryResponseProcessor<>(
                        Arrays.asList("R1", "R2"),
                        Arrays.asList("TRUE", "TRUE"),
                        new MockMultiOutputFactory(), false)
                , Arrays.asList(
                        new HashSet<>(Arrays.asList("1", "0")),
                        new HashSet<>(Arrays.asList("1", "1")),
                        new HashSet<>(Arrays.asList("0", "0")),
                        new HashSet<>(Arrays.asList("1", "0")),
                        new HashSet<>(Arrays.asList("0", "1")),
                        new HashSet<>(Arrays.asList("0", "1")))
                , MockMultiOutput::getNameSet);

        doTest(new BinaryResponseProcessor<>("R1", "TRUE", new MockOutputFactory())
                , Arrays.asList("1", "1", "0", "1", "0", "0")
                , mo -> mo.label);
    }

    @Test
    public void fieldTest() {

        doTest(new FieldResponseProcessor<>(
                        Arrays.asList("R1", "R2"),
                        Arrays.asList("FALSE", "FALSE"),
                        new MockMultiOutputFactory(), true)
                , multiTruthValues
                , MockMultiOutput::getNameSet);

        doTest(new FieldResponseProcessor<>(
                        Arrays.asList("R1", "R2"),
                        Arrays.asList("FALSE", "FALSE"),
                        new MockMultiOutputFactory(), false)
                , Arrays.asList(
                        new HashSet<>(Arrays.asList("TRUE", "FALSE")),
                        new HashSet<>(Arrays.asList("TRUE", "TRUE")),
                        new HashSet<>(Arrays.asList("FALSE", "FALSE")),
                        new HashSet<>(Arrays.asList("TRUE", "FALSE")),
                        new HashSet<>(Arrays.asList("FALSE", "TRUE")),
                        new HashSet<>(Arrays.asList("FALSE", "TRUE")))
                , MockMultiOutput::getNameSet);

        doTest(new FieldResponseProcessor<>(
                        "R1",
                        "FALSE",
                        new MockOutputFactory())
                , Arrays.asList("TRUE", "TRUE", "FALSE", "TRUE", "FALSE", "FALSE")
                , mo -> mo.label);
    }

    @Test
    public void quartileTest() {
        // we're using column "C" as output
        Quartile quartile = new Quartile(5, 3, 8);
        doTest(new QuartileResponseProcessor<>(
                        Collections.singletonList("C"),
                        Collections.singletonList(quartile),
                        new MockOutputFactory())
                , Arrays.asList("C:first", "C:third", "C:third", "C:first", "C:second", "C:second"),
                mo -> mo.label);
    }
}
