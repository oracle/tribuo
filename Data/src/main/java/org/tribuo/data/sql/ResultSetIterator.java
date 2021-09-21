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

package org.tribuo.data.sql;

import com.oracle.labs.mlrg.olcut.util.IOSpliterator;
import org.tribuo.data.columnar.ColumnarIterator;

import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An iterator over a ResultSet returned from JDBC.
 */
public class ResultSetIterator extends ColumnarIterator {
    private static final Logger logger = Logger.getLogger(ResultSetIterator.class.getName());

    private final ResultSet resultSet;

    private int rowNum = 0;

    /**
     * Construct a result set iterator over the supplied result set.
     * @param rs The result set.
     * @throws SQLException If the result set cannot be inspected.
     */
    public ResultSetIterator(ResultSet rs) throws SQLException {
        resultSet = rs;
        ResultSetMetaData rsm = resultSet.getMetaData();
        fields = new ArrayList<>();
        for(int i=1; i <= rsm.getColumnCount(); i++) {
            fields.add(rsm.getColumnName(i));
        }
    }

    /**
     * Constructs a result set iterator over the supplied result set using the specified fetch buffer size.
     * @param rs The result set.
     * @param fetchSize The fetch size.
     * @throws SQLException If the result set cannot be inspected.
     */
    public ResultSetIterator(ResultSet rs, int fetchSize) throws SQLException {
        super(IOSpliterator.DEFAULT_CHARACTERISTICS, fetchSize, Long.MAX_VALUE);
        resultSet = rs;
        ResultSetMetaData rsm = resultSet.getMetaData();
        fields = new ArrayList<>();
        for(int i=1; i <= rsm.getColumnCount(); i++) {
            fields.add(rsm.getColumnName(i));
        }
    }

    @Override
    protected Optional<Row> getRow() {
        try {
            if(!resultSet.isClosed() && resultSet.next()) {
                Map<String, String> rowMap = new HashMap<>();
                for(int i=0; i < fields.size(); i++) {
                    Object obj = null;
                    try {
                        obj = resultSet.getObject(i + 1);
                    } catch (SQLException e) {
                        logger.log(Level.SEVERE, "Missing object at index: " + (i + 1), e);
                    }
                    rowMap.put(fields.get(i), obj == null ? "" : obj.toString());
                }
                rowNum++;
                if (rowNum % 50_000 == 0) {
                    logger.info(String.format("Iterated over %d rows", rowNum));
                }
                return Optional.of(new Row(rowNum, fields, rowMap));
            } else {
                if(!resultSet.isClosed()) {
                    resultSet.close();
                }
                return Optional.empty();
            }
        } catch (SQLException e) {
            try {
                resultSet.close();
            } catch (SQLException e2) {
                logger.log(Level.WARNING, "Error closing ResultSet inside another error", e2);
            }
            throw new IllegalStateException("Error while reading from SQL at row " + rowNum, e);
        }
    }
}
