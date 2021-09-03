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

import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.RFC4180ParserBuilder;
import com.opencsv.exceptions.CsvValidationException;
import org.tribuo.data.columnar.ColumnarIterator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An iterator over a CSV file.
 */
public class CSVIterator extends ColumnarIterator implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(CSVIterator.class.getName());

    /**
     * Default separator character.
     */
    public final static char SEPARATOR = ',';

    /**
     * Default quote character.
     */
    public final static char QUOTE = '"';

    private final CSVReader reader;

    // Used as the row index
    private int recordNum = 0;

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
        this(rdr, separator, quote, Collections.emptyList());
    }

    /**
     * Builds a CSVIterator for the supplied URI. Defaults to {@link CSVIterator#SEPARATOR} for the separator
     * and {@link CSVIterator#QUOTE} for the quote.
     * @param dataFile The source to read.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public CSVIterator(URI dataFile) throws IOException {
        this(new InputStreamReader(Files.newInputStream(Paths.get(dataFile)), StandardCharsets.UTF_8));
    }

    /**
     * Builds a CSVIterator for the supplied URI.
     * @param dataFile The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public CSVIterator(URI dataFile, char separator, char quote) throws IOException {
        this(new InputStreamReader(Files.newInputStream(Paths.get(dataFile)), StandardCharsets.UTF_8), separator, quote);
    }

    /**
     * Builds a CSVIterator for the supplied URI. If headers is null or an empty array, read the headers from the csv file.
     * @param dataFile The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @param fields The headers to use.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public CSVIterator(URI dataFile, char separator, char quote, String[] fields) throws IOException {
        this(new InputStreamReader(Files.newInputStream(Paths.get(dataFile)), StandardCharsets.UTF_8), separator, quote, Arrays.asList(fields));
    }

    /**
     * Builds a CSVIterator for the supplied URI.
     * @param dataFile The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @param fields The headers to use.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public CSVIterator(URI dataFile, char separator, char quote, List<String> fields) throws IOException {
        this(new InputStreamReader(Files.newInputStream(Paths.get(dataFile)), StandardCharsets.UTF_8), separator, quote, fields);
    }

    /**
     * Builds a CSVIterator for the supplied Reader. If headers is null or an empty array, read the headers from the csv file.
     * @param rdr The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @param fields The headers to use.
     */
    public CSVIterator(Reader rdr, char separator, char quote, String[] fields) {
        this(rdr, separator, quote, fields == null ? null : Arrays.asList(fields));
    }

    /**
     * Builds a CSVIterator for the supplied Reader. If headers is null or an empty list, read the headers from the csv file.
     * @param rdr The source to read.
     * @param separator The separator character to use.
     * @param quote The quote character to use.
     * @param fields The headers to use.
     */
    public CSVIterator(Reader rdr, char separator, char quote, List<String> fields) {
        try {
            // If someone hands us a BufferedReader, then we'll double buffer it here.
            Reader bomRemoved = new BufferedReader(CSVDataSource.removeBOM(rdr));
            reader = new CSVReaderBuilder(bomRemoved).withCSVParser(new RFC4180ParserBuilder().withSeparator(separator).withQuoteChar(quote).build()).build();
            try {
                if (fields == null || fields.isEmpty()) {
                    String[] inducedHeader = reader.readNext();
                    if(inducedHeader == null) {
                        logger.warning("Given an empty CSV");
                    } else {
                        this.fields = Collections.unmodifiableList(Arrays.asList(inducedHeader));
                    }
                } else {
                    this.fields = Collections.unmodifiableList(fields);
                }
            } catch (CsvValidationException | IOException e) {
                try {
                    reader.close();
                } catch (IOException e2) {
                    logger.log(Level.WARNING, "Error closing reader in another error", e2);
                }
                throw new IllegalArgumentException("Error reading file caused by: " + e.getMessage());
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Error reading file caused by: " + e.getMessage());
        }
    }

    /**
     * Zips together the headers and the line into a Map.
     * @param headers The field headers.
     * @param line The extracted line.
     * @param rowNum The row number used for error messages.
     * @return A map from header to value.
     */
    private static Map<String,String> zip(List<String> headers, String[] line, long rowNum) {
        if (headers.size() != line.length) {
            throw new IllegalArgumentException("On row " + rowNum + " headers has " + headers.size() + " elements, current line has " + line.length + " elements.");
        }

        Map<String,String> map = new HashMap<>();
        for (int i = 0; i < headers.size(); i++) {
            map.put(headers.get(i),line[i]);
        }
        return map;
    }

    @Override
    protected Optional<Row> getRow() {
        try {
            String[] rawRow = reader.readNext();
            if(rawRow != null) {
                if(reader.getRecordsRead() % 50_000 == 0) {
                    logger.info(String.format("Read %d records on %d lines", reader.getRecordsRead(), reader.getLinesRead()));
                }
                while (rawRow != null && rawRow.length == 1 && rawRow[0].isEmpty()) {
                    // Found an extraneous newline in the csv file
                    logger.warning("Ignoring extra newline at line " + reader.getLinesRead());
                    rawRow = reader.readNext();
                }
                if (rawRow == null) {
                    try {
                        reader.close();
                    } catch (IOException e) {
                        logger.log(Level.WARNING, "Error closing reader at end of file", e);
                    }
                    return Optional.empty();
                }

                // Note this is intentionally recordNum++ as we count records from 0.
                return Optional.of(new Row(recordNum++,
                        fields,
                        zip(fields, rawRow, reader.getRecordsRead())));
            } else {
                try {
                    reader.close();
                } catch (IOException e) {
                    logger.log(Level.WARNING, "Error closing reader at end of file", e);
                }
                return Optional.empty();
            }
        } catch (CsvValidationException | IOException e) {
            long linesRead = reader.getLinesRead();
            long recordsRead = reader.getRecordsRead();
            try {
                reader.close();
            } catch (IOException e2) {
                logger.log(Level.WARNING, "Error closing reader in another error", e2);
            }
            throw new IllegalArgumentException(String.format("Error reading CSV on record %d, row %d", recordsRead, linesRead), e);
        }
    }

    @Override
    public void close() throws IOException{
        if(reader != null) {
            reader.close();
        }
    }
}
