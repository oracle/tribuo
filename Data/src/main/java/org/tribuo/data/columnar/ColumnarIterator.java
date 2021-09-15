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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.util.IOSpliterator;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * An abstract class for iterators that read data in to a columnar format, usually from a file of some kind. Subclasses
 * handle how to format data from that file.
 * <p>
 * Note: the {@code fields} field must be set in the constructor of implementing classes.
 */
public abstract class ColumnarIterator extends IOSpliterator<ColumnarIterator.Row> implements Iterator<ColumnarIterator.Row> {
    private static final Logger logger = Logger.getLogger(ColumnarIterator.class.getName());

    /**
     * The column headers for this iterator.
     */
    protected List<String> fields;
    /**
     * The current row.
     */
    protected Optional<Row> currentRow = Optional.empty();

    /**
     * A representation of a row of untyped data from a columnar data source. In addition to its row data it stores a
     * canonical field list and its index (from 0) in the original data source. It should be immutable once constructed.
     * It is the responsibility of the implementor of ColumnarIterator to ensure that the passed field list is immutable.
     */
    public static class Row {

        private final long index;
        private final List<String> fields;
        private final Map<String, String> rowData;

        /**
         * Constructs a row from a columnar source.
         * @param index The row index.
         * @param fields The field names, assumed to be immutable.
         * @param rowData The row data.
         */
        public Row(long index, List<String> fields, Map<String, String> rowData) {
            this.index = index;
            this.fields = fields;
            this.rowData = Collections.unmodifiableMap(rowData);
        }

        /**
         * Gets the field headers.
         * @return The field headers.
         */
        public List<String> getFields() {
            return fields;
        }

        /**
         * Gets the row index.
         * @return The row index.
         */
        public long getIndex() {
            return index;
        }

        /**
         * Gets the row data.
         * @return The row data.
         */
        public Map<String, String> getRowData() {
            return rowData;
        }

        @Override
        public String toString() {
            return "Row(index=" + index + ", fields=" + fields.toString() + ", rowData=" + rowData.toString() + ")";
        }
    }


    /**
     * Constructs a ColumnarIterator wrapped around a buffering spliterator.
     * <p>
     * Note when using this constructor you must set the {@code fields} field to
     * the appropriate value after you've called super(). It must be immutable.
     */
    protected ColumnarIterator() {
        super();
    }

    /**
     * Constructs a ColumnarIterator wrapped around a buffering spliterator.
     * <p>
     * Note when using this constructor you must set the {@code fields} field to
     * the appropriate value after you've called super(). It must be immutable.
     * @param characteristics The spliterator characteristics.
     * @param batchsize The buffer size.
     * @param estimatedSize The estimated size of this iterator.
     */
    protected ColumnarIterator(int characteristics, int batchsize, long estimatedSize) {
        super(characteristics, batchsize, estimatedSize);
    }

    /**
     * The immutable list of field names.
     * @return The field names.
     */
    public List<String> getFields() {
        return fields;
    }

    @Override
    public boolean hasNext() {
        if (currentRow.isPresent()) {
            return true;
        } else {
            currentRow = getRow();
            return currentRow.isPresent();
        }
    }

    @Override
    public Row next() {
        if (hasNext()) {
            Row r = currentRow.get();
            currentRow = Optional.empty();
            return r;
        } else {
            throw new NoSuchElementException();
        }
    }

    @Override
    public boolean tryAdvance(Consumer<? super Row> action) {
        if (hasNext()) {
            action.accept(next());
            return true;
        } else {
            return false;
        }
    }

    @Override
    public void forEachRemaining(Consumer<? super Row> action) {
        while (hasNext()) {
            action.accept(next());
        }
    }

    /**
     * Returns the next row of data based on internal state stored by the implementor, or {@link Optional#empty()}
     * if there is no more data.
     * @return The next row of data or None.
     */
    protected abstract Optional<Row> getRow();
}
