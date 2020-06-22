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

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.RFC4180ParserBuilder;
import com.opencsv.exceptions.CsvValidationException;
import org.tribuo.data.columnar.FieldNames;

import java.io.Closeable;
import java.io.IOException;
import java.io.Reader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.logging.Logger;

/**
 * An iterator for a CSV file that makes it compatible with {@link org.tribuo.data.columnar.RowProcessor}.
 */
public class CSVIterator implements Iterator<Map<String,String>>, FieldNames, Closeable {
    private static final Logger logger = Logger.getLogger(CSVIterator.class.getName());

    public final static char SEPARATOR = ',';
    public final static char QUOTE = '"';

    private Map<String,String> curEntry;
    private final CSVReader reader;
    private final String[] headers;
    private int rowNum;

    /**
     * Builds a CSVIterator for the supplied Reader. Defaults to {@link CSVIterator#SEPARATOR} for the separator
     * and {@link CSVIterator#QUOTE} for the quote.
     * @param rdr The source to read.
     */
    public CSVIterator(Reader rdr) {
        this(rdr, SEPARATOR, QUOTE);
    }

    /**
     * Builds a CSVIterator for the supplied Reader.
     * @param rdr The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     */
    public CSVIterator(Reader rdr, char separator, char quote) {
        this(rdr,separator,quote,null);
    }

    /**
     * Builds a CSVIterator for the supplied Reader. If headers is null, read the headers from the csv file.
     * @param rdr The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @param headers The headers to use. Set to null to read the headers from the CSV file.
     */
    public CSVIterator(Reader rdr, char separator, char quote, String[] headers) {
        rowNum = 0;
        reader = new CSVReaderBuilder(rdr).withCSVParser(new RFC4180ParserBuilder().withSeparator(separator).withQuoteChar(quote).build()).build();
        try {
            if (headers == null) {
                this.headers = reader.readNext();
                if (this.headers == null) {
                    throw new IllegalStateException("CSV file had no header row, and none was provided");
                }
                rowNum++;
            } else {
                this.headers = Arrays.copyOf(headers,headers.length);
            }
            String[] firstRow = reader.readNext();
            if(firstRow != null) {
                curEntry = zip(this.headers, firstRow, rowNum);
            } else {
                logger.warning("CSV file had no data");
            }
        } catch (CsvValidationException | IOException e) {
            throw new NoSuchElementException("Error reading file caused by: " + e.getMessage());
        }
    }

    /**
     * Builds a CSVIterator for the supplied URI. Defaults to {@link CSVIterator#SEPARATOR} for the separator
     * and {@link CSVIterator#QUOTE} for the quote.
     * @param dataFile The source to read.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public CSVIterator(URI dataFile) throws IOException {
        this(Files.newBufferedReader(Paths.get(dataFile)));
    }

    /**
     * Builds a CSVIterator for the supplied URI.
     * @param dataFile The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public CSVIterator(URI dataFile, char separator, char quote) throws IOException {
        this(Files.newBufferedReader(Paths.get(dataFile)), separator, quote);
    }

    @Override
    public String[] fields() {
        return headers;
    }

    @Override
    public boolean hasNext() {
        return curEntry != null;
    }

    @Override
    public Map<String, String> next() {
        String[] row;
        try {
            row = reader.readNext();
            if ((row == null) || (row.length < 2)) {
                reader.close();
            }
        } catch (CsvValidationException | IOException e) {
            throw new NoSuchElementException("Error reading data - " + e.getMessage());
        }
        Map<String, String> result = curEntry;
        rowNum++;
        curEntry = row == null ? null : zip(headers, row, rowNum);
        return result;
    }

    /**
     * Get the current row number.
     * @return The row number.
     */
    public int getRowNum() {
        return rowNum;
    }

    private static Map<String,String> zip(String[] headers, String[] line, int rowNum) {
        if (headers.length != line.length) {
            throw new IllegalArgumentException("On row " + rowNum + " headers has " + headers.length + " elements, current line has " + line.length + " elements.");
        }

        Map<String,String> map = new HashMap<>();
        for (int i = 0; i < headers.length; i++) {
            map.put(headers[i],line[i]);
        }
        return map;
    }

    @Override
    public void close() throws IOException {
        reader.close();
    }
}



