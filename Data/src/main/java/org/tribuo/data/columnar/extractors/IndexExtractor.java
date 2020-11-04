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
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.data.columnar.ColumnarIterator;
import org.tribuo.data.columnar.FieldExtractor;

import java.util.Optional;

/**
 * An Extractor with special casing for loading the index from a Row.
 * The index is written out as a Long.
 * <p>
 * This is the row wise count, i.e., the number of examples that the data
 * source has processed, rather than anything extracted from the data.
 */
public class IndexExtractor implements FieldExtractor<Long> {

    @Config(description = "The metadata key to emit, defaults to Example.NAME")
    private String metadataName = Example.NAME;

    /**
     * Extracts the index, writing to the supplied metadata field name.
     * @param metadataName The metadata field to write to.
     */
    public IndexExtractor(String metadataName) {
        this.metadataName = metadataName;
    }

    /**
     * Extracts the index writing to the default metadata field name {@link Example#NAME}.
     */
    public IndexExtractor() {}

    @Override
    public String getMetadataName() {
        return metadataName;
    }

    @Override
    public Class<Long> getValueType() {
        return Long.class;
    }

    @Override
    public Optional<Long> extract(ColumnarIterator.Row row) {
        return row.getIndex() == -1 ? Optional.empty() : Optional.of(row.getIndex());
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "IndexExtractor");
    }
}
