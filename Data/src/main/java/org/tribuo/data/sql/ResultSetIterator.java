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

import org.tribuo.data.columnar.FieldNames;

import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An iterator over a SQL result set that makes it compatible with {@link org.tribuo.data.columnar.RowProcessor}.
 */
public class ResultSetIterator implements Iterator<Map<String, String>>, FieldNames, AutoCloseable {

    private static final Logger logger = Logger.getLogger(ResultSetIterator.class.getName());

    private ResultSet resultSet;
    private String[] fields;
    private boolean hasNext;

    private int rowNum = 0;

    public ResultSetIterator(ResultSet rs) throws SQLException {
        resultSet = rs;
        ResultSetMetaData rsm = resultSet.getMetaData();
        fields = new String[rsm.getColumnCount()];
        for(int i=1; i <= rsm.getColumnCount(); i++) {
            fields[i-1] = rsm.getColumnName(i);
        }
        logger.info("Iterating over result set with fields: " + String.join(", ", fields));
        try {
            hasNext = resultSet.next();
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Exception", e);
            hasNext = false;
        }
    }

    @Override
    public String[] fields() {
        return fields;
    }

    @Override
    public boolean hasNext() {
        if(!hasNext) {
            logger.info(String.format("Finished iterating over %d rows", rowNum));
            try {
                resultSet.close();
            } catch (SQLException e) {
                logger.log(Level.WARNING, "Error closing statement", e);
            }
        }
        return hasNext;
    }

    @Override
    public Map<String, String> next() {
        Map<String, String> currentRow = new HashMap<>();
        for(int i=0; i < fields.length; i++) {
            Object obj = null;
            try {
                obj = resultSet.getObject(i + 1);
            } catch (SQLException e) {
                logger.log(Level.SEVERE, "Missing object at index: " + (i + 1), e);
            }
            String resString = obj == null ? "" : obj.toString();
            currentRow.put(fields[i], resString);
        }

        rowNum++;
        if (rowNum % 50_000 == 0) {
            logger.warning(String.format("Iterated over %d rows", rowNum));
        }

        try {
            hasNext = resultSet.next();
        } catch (SQLException e) {
            logger.log(Level.SEVERE, "Exception", e);
            hasNext = false;
        }
        if(currentRow.isEmpty()) {
            throw new NoSuchElementException();
        }
        return currentRow;
    }

    @Override
    public void close() throws SQLException {
        resultSet.close();
    }
}

