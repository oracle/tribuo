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

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

import java.util.Optional;
import java.util.logging.Logger;

/**
 * Extracts the field value and converts it to a float.
 * <p>
 * Returns an empty optional if the value failed to parse.
 */
public class FloatExtractor extends SimpleFieldExtractor<Float> {

    private static final Logger logger = Logger.getLogger(FloatExtractor.class.getName());

    /**
     * Extracts a float value from the supplied field name.
     * Writes the metadata out using the field name as the key.
     * @param fieldName The field name to inspect.
     */
    public FloatExtractor(String fieldName) {
        super(fieldName);
    }

    /**
     * Extracts a float value from the supplied field name.
     * Writes the metadata out using the metadataName as the key.
     * @param fieldName The field name to inspect.
     * @param metadataName The metadata name to emit.
     */
    public FloatExtractor(String fieldName, String metadataName) {
        super(fieldName, metadataName);
    }

    /**
     * For olcut.
     */
    private FloatExtractor() {}

    @Override
    public Class<Float> getValueType() {
        return Float.class;
    }

    @Override
    protected Optional<Float> extractField(String value) {
        try {
            float f = Float.parseFloat(value);
            return Optional.of(f);
        } catch (NumberFormatException e) {
            logger.warning("Failed to parse value for field " + fieldName + ", expected a float, got " + value);
        }
        return Optional.empty();
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldExtractor");
    }
}
