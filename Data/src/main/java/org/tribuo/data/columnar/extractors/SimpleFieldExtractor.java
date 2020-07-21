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

package org.tribuo.data.columnar.extractors;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.data.columnar.ColumnarIterator;
import org.tribuo.data.columnar.FieldExtractor;

import java.util.Optional;
import java.util.logging.Logger;

/**
 * Extracts a value from a single field to be placed in an {@link org.tribuo.Example}'s metadata field.
 *
 */
public abstract class SimpleFieldExtractor<T> implements FieldExtractor<T> {

    private static final Logger logger = Logger.getLogger(SimpleFieldExtractor.class.getName());

    @Config(mandatory = true,description="The field name to read.")
    protected String fieldName;

    @Config(description="The metadata key to emit, defaults to field name if unpopulated")
    protected String metadataName;

    protected SimpleFieldExtractor(String fieldName) {
        this(fieldName, fieldName);
    }

    protected SimpleFieldExtractor(String fieldName, String metadataName) {
        this.fieldName = fieldName;
        this.metadataName = metadataName;
    }

    protected SimpleFieldExtractor() {}

    @Override
    public void postConfig() {
        if (metadataName == null || metadataName.isEmpty()) {
            metadataName = fieldName;
        }
    }

    /**
     * Gets the field name this extractor operates on.
     * @return The field name.
     */
    public String getFieldName() {
        return fieldName;
    }

    /**
     * Gets the metadata key name. This is the key into which this value will be written in an {@link org.tribuo.Example}
     * if it is given to {@link org.tribuo.data.columnar.RowProcessor#metadataExtractors}.
     *
     * Defaults to the field name.
     * @return The metadata key name.
     */
    @Override
    public String getMetadataName() {
        return metadataName;
    }

    protected abstract Optional<T> extractField(String fieldValue);

    @Override
    public Optional<T> extract(ColumnarIterator.Row row) {
        if(row.getRowData().containsKey(getFieldName())) {
            return extractField(row.getRowData().get(getFieldName()));
        } else {
            logger.warning("Row was missing expected field " + getFieldName());
            return Optional.empty();
        }
    }

    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(fieldName=" + fieldName + ", metadataName=" + metadataName + ")";
    }
}
