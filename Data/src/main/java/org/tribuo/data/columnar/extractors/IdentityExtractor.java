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
import org.tribuo.data.columnar.FieldExtractor;

import java.util.Optional;
import java.util.logging.Logger;

/**
 * Extracts the field value and emits it as a String.
 */
public class IdentityExtractor implements FieldExtractor<String> {
    private static final Logger logger = Logger.getLogger(IdentityExtractor.class.getName());

    @Config(mandatory = true, description = "The field name to read.")
    private String fieldName;

    @Config(description = "The metadata key to emit, defaults to field name if unpopulated")
    private String metadataName;

    public IdentityExtractor(String fieldName, String metadataName) {
        this.fieldName = fieldName;
        this.metadataName = metadataName;
    }

    /**
     * For olcut.
     */
    private IdentityExtractor() {}

    @Override
    public void postConfig() {
        if (metadataName == null || metadataName.isEmpty()) {
            metadataName = fieldName;
        }
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public String getMetadataName() {
        return metadataName;
    }

    @Override
    public Class<String> getValueType() {
        return String.class;
    }

    @Override
    public Optional<String> extract(String value) {
        if (value == null || value.isEmpty()) {
            return Optional.empty();
        } else {
            return Optional.of(value);
        }
    }

    @Override
    public String toString() {
        return "IdentityExtractor(fieldName="+fieldName+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldExtractor");
    }
}
