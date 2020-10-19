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
 * Extracts the field value and emits it as a String.
 */
public class IdentityExtractor extends SimpleFieldExtractor<String> {
    private static final Logger logger = Logger.getLogger(IdentityExtractor.class.getName());

    /**
     * Extracts the String value from the supplied field.
     * Writes the metadata out using the field name as the key.
     * @param fieldName The field name to inspect.
     */
    public IdentityExtractor(String fieldName) {
        super(fieldName);
    }

    /**
     * Extracts the String value from the supplied field.
     * Writes the metadata out using the metadataName as the key.
     * @param fieldName The field name to inspect.
     * @param metadataName The metadata name to emit.
     */
    public IdentityExtractor(String fieldName, String metadataName) {
        super(fieldName, metadataName);
    }

    /**
     * For olcut.
     */
    private IdentityExtractor() {}

    @Override
    public Class<String> getValueType() {
        return String.class;
    }

    @Override
    protected Optional<String> extractField(String value) {
        if (value == null || value.isEmpty()) {
            return Optional.empty();
        } else {
            return Optional.of(value);
        }
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldExtractor");
    }
}
