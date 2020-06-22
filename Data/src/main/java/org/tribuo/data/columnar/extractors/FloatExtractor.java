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
 * Extracts the field value and converts it to a float.
 * <p>
 * Returns an empty optional if the value failed to parse.
 */
public class FloatExtractor implements FieldExtractor<Float> {

    private static final Logger logger = Logger.getLogger(FloatExtractor.class.getName());

    @Config(mandatory = true, description = "The field name to read.")
    private String fieldName;

    @Config(description = "The metadata key to emit, defaults to field name if unpopulated")
    private String metadataName;

    public FloatExtractor(String fieldName) {
        this(fieldName, WEIGHT_NAME);
    }

    public FloatExtractor(String fieldName, String metadataName) {
        this.fieldName = fieldName;
        this.metadataName = metadataName;
    }

    /**
     * For olcut.
     */
    private FloatExtractor() {}

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
    public Class<Float> getValueType() {
        return Float.class;
    }

    @Override
    public Optional<Float> extract(String value) {
        try {
            float f = Float.parseFloat(value);
            return Optional.of(f);
        } catch (NumberFormatException e) {
            logger.warning("Failed to parse value for field " + fieldName + ", expected a float, got " + value);
        }
        return Optional.empty();
    }

    @Override
    public String toString() {
        return "FloatExtractor(fieldName="+fieldName+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldExtractor");
    }
}
