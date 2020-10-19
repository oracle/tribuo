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

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import java.util.Optional;

/**
 * Extracts a value from a field to be placed in an {@link org.tribuo.Example}'s metadata field.
 * Principally used to pull out Strings and floats for example names and weights respectively.
 */
public interface FieldExtractor<T> extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Gets the metadata key name. This is the key into which this value will be written in an {@link org.tribuo.Example}
     * if it is given to {@link RowProcessor}.
     * @return The metadata key name.
     */
    public String getMetadataName();

    /**
     * Gets the class of the value produced by this extractor.
     * @return The class of the value.
     */
    public Class<T> getValueType();

    /**
     * Returns Optional which is filled if extraction succeeded.
     * @param row The raw row from which to extract a value
     * @return A value.
     */
    Optional<T> extract(ColumnarIterator.Row row);
}
