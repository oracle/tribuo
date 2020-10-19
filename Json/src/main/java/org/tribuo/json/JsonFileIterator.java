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

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.data.columnar.ColumnarIterator;

import java.io.IOException;
import java.io.Reader;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An iterator for JSON format files converting them into a format suitable for
 * {@link org.tribuo.data.columnar.RowProcessor}.
 */
public class JsonFileIterator extends ColumnarIterator implements AutoCloseable {
    private static final Logger logger = Logger.getLogger(JsonFileIterator.class.getName());

    private final JsonParser parser;
    private final Iterator<JsonNode> nodeIterator;
    private int rowNum = 0;

    /**
     * Builds a JsonFileIterator for the supplied Reader.
     * @param reader The source to read.
     */
    public JsonFileIterator(Reader reader) {
        JsonFactory jsonFactory = new JsonFactory();
        //noinspection OverlyBroadCatchBlock
        try {
            parser = jsonFactory.createParser(reader);
            parser.setCodec(new ObjectMapper());
            JsonNode jsonNode = parser.readValueAsTree();
            if (jsonNode.isArray()) {
                ArrayNode node = (ArrayNode) jsonNode;
                nodeIterator = node.elements();
                if (nodeIterator.hasNext()) {
                    JsonNode curNode = nodeIterator.next();
                    if (curNode instanceof ObjectNode) {
                        Map<String, String> curEntry = JsonUtil.convertToMap((ObjectNode)curNode);
                        List<String> headerList = new ArrayList<>(curEntry.keySet());
                        Collections.sort(headerList);
                        fields = headerList;
                        currentRow = Optional.of(new Row(rowNum, fields, curEntry));
                        rowNum++;
                    } else {
                        throw new IllegalStateException("Expected an array of JSON objects but found '" + curNode.asText() + "'");
                    }
                } else {
                    throw new IllegalStateException("No elements found in JSON array");
                }
            } else {
                throw new IllegalStateException("JSON array not found when reading file");
            }
        } catch (IOException e) {
            throw new IllegalStateException("Error reading json file caused by: " + e.getMessage());
        }
    }

    /**
     * Builds a CSVIterator for the supplied URI.
     * @param dataFile The source to read.
     * @throws IOException thrown if the file is not readable in some way.
     */
    public JsonFileIterator(URI dataFile) throws IOException {
        this(Files.newBufferedReader(Paths.get(dataFile)));
    }

    @Override
    protected Optional<Row> getRow() {
        // row is initially populated in the constructor
        if (nodeIterator.hasNext()) {
            JsonNode next = nodeIterator.next();
            if (next instanceof ObjectNode) {
                Row row = new Row(rowNum, fields, JsonUtil.convertToMap((ObjectNode)next));
                rowNum++;
                return Optional.of(row);
            } else {
                logger.warning("Unexpected node found, expected ObjectNode, found '" + next.asText() + '"');
                return Optional.empty();
            }
        } else {
            try {
                parser.close();
            } catch (IOException e) {
                logger.log(Level.WARNING, "Error closing reader at end of file", e);
            }
            return Optional.empty();
        }
    }

    @Override
    public void close() throws IOException {
        parser.close();
    }
}
