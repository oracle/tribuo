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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.tribuo.data.columnar.ColumnarIterator;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class CSVIteratorTest {

    private URI path;
    private URI noHeaderPath;
    private URI quotePath;
    private URI tsvPath;
    private URI doubleLineBreak;
    private List<ColumnarIterator.Row> pathReference;
    private List<String> headers;

    @BeforeEach
    public void setUp() throws URISyntaxException {
        path = getClass().getResource("/org/tribuo/data/csv/test.csv").toURI();
        noHeaderPath = getClass().getResource("/org/tribuo/data/csv/test-noheader.csv").toURI();
        quotePath = getClass().getResource("/org/tribuo/data/csv/testQuote.csv").toURI();
        tsvPath = getClass().getResource("/org/tribuo/data/csv/testQuote.tsv").toURI();
        doubleLineBreak = getClass().getResource("/org/tribuo/data/csv/test-double-line-break.csv").toURI();

        headers = Arrays.asList("A B C D RESPONSE".split(" "));
        pathReference = new ArrayList<>();
        Map<String, String> rVals = new HashMap<>();
        rVals.put("A", "1");
        rVals.put("B", "2");
        rVals.put("C", "3");
        rVals.put("D", "4");
        rVals.put("RESPONSE", "monkey");
        pathReference.add(new ColumnarIterator.Row(0, headers, rVals));
        rVals = new HashMap<>();
        rVals.put("A", "2");
        rVals.put("B", "5");
        rVals.put("C", "3");
        rVals.put("D", "4");
        rVals.put("RESPONSE", "monkey");
        pathReference.add(new ColumnarIterator.Row(1, headers, rVals));
        rVals = new HashMap<>();
        rVals.put("A", "1");
        rVals.put("B", "2");
        rVals.put("C", "5");
        rVals.put("D", "9");
        rVals.put("RESPONSE", "baboon");
        pathReference.add(new ColumnarIterator.Row(2, headers, rVals));
        rVals = new HashMap<>();
        rVals.put("A", "3");
        rVals.put("B", "5");
        rVals.put("C", "8");
        rVals.put("D", "4");
        rVals.put("RESPONSE", "monkey");
        pathReference.add(new ColumnarIterator.Row(3, headers, rVals));
        rVals = new HashMap<>();
        rVals.put("A", "6");
        rVals.put("B", "7");
        rVals.put("C", "8");
        rVals.put("D", "9");
        rVals.put("RESPONSE", "baboon");
        pathReference.add(new ColumnarIterator.Row(4, headers, rVals));
        rVals = new HashMap<>();
        rVals.put("A", "0");
        rVals.put("B", "7");
        rVals.put("C", "8");
        rVals.put("D", "9");
        rVals.put("RESPONSE", "baboon");
        pathReference.add(new ColumnarIterator.Row(5, headers, rVals));
    }

    @Test
    public void testCsvReadingCorrectly() throws IOException {
        CSVIterator iter = new CSVIterator(path);
        for(int i=0; i < pathReference.size();i++) {
            ColumnarIterator.Row iterRow = iter.next();
            ColumnarIterator.Row refRow = pathReference.get(i);
            assertEquals(refRow.getIndex(), iterRow.getIndex(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getFields(), iterRow.getFields(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getRowData(), iterRow.getRowData(), "Failure on row " + i + " of " + path.toString());
        }
        assertFalse(iter.hasNext(), "Iterator should be empty after reading");
    }

    @Test
    public void testQuotedCsvReadingCorrectly() throws IOException {
        CSVIterator iter = new CSVIterator(quotePath);
        for(int i=0; i < pathReference.size();i++) {
            ColumnarIterator.Row iterRow = iter.next();
            ColumnarIterator.Row refRow = pathReference.get(i);
            assertEquals(refRow.getIndex(), iterRow.getIndex(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getFields(), iterRow.getFields(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getRowData(), iterRow.getRowData(), "Failure on row " + i + " of " + path.toString());
        }
        assertFalse(iter.hasNext(), "Iterator should be empty after reading");
    }

    @Test
    public void testNoHeaderReadingCorrectly() throws IOException {
        CSVIterator iter = new CSVIterator(noHeaderPath, CSVIterator.SEPARATOR, CSVIterator.QUOTE, headers);
        for(int i=0; i < pathReference.size();i++) {
            ColumnarIterator.Row iterRow = iter.next();
            ColumnarIterator.Row refRow = pathReference.get(i);
            assertEquals(refRow.getIndex(), iterRow.getIndex(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getFields(), iterRow.getFields(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getRowData(), iterRow.getRowData(), "Failure on row " + i + " of " + path.toString());
        }
        assertFalse(iter.hasNext(), "Iterator should be empty after reading");
    }

    @Test
    public void testQuotedTsvReadingCorrectly() throws IOException {
        CSVIterator iter = new CSVIterator(tsvPath, '\t', '|');
        for(int i=0; i < pathReference.size();i++) {
            ColumnarIterator.Row iterRow = iter.next();
            ColumnarIterator.Row refRow = pathReference.get(i);
            assertEquals(refRow.getIndex(), iterRow.getIndex(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getFields(), iterRow.getFields(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getRowData(), iterRow.getRowData(), "Failure on row " + i + " of " + path.toString());
        }
        assertFalse(iter.hasNext(), "Iterator should be empty after reading");
    }

    @Test
    public void testDoubleLineBreakReadingCorrectly() throws IOException {
        CSVIterator iter = new CSVIterator(doubleLineBreak);
        for(int i=0; i < pathReference.size();i++) {
            ColumnarIterator.Row iterRow = iter.next();
            ColumnarIterator.Row refRow = pathReference.get(i);
            assertEquals(refRow.getIndex(), iterRow.getIndex(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getFields(), iterRow.getFields(), "Failure on row " + i + " of " + path.toString());
            assertEquals(refRow.getRowData(), iterRow.getRowData(), "Failure on row " + i + " of " + path.toString());
        }
        assertFalse(iter.hasNext(), "Iterator should be empty after reading");
    }

}
