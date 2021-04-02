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

package org.tribuo.data.columnar.processors.field;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FieldProcessor;

import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Processes a column that contains a real value. The name of the feature
 * will be the name given for the column and the value will be the value in the 
 * column. This processor returns all doubles that can be parsed by {@link Double#parseDouble(String)}
 * including zeros, and so will emit zero valued features.
 * <p>
 * Returns an empty list if the value failed to parse or was empty.
 */
public class DoubleFieldProcessor implements FieldProcessor {
    
    private static final Logger logger = Logger.getLogger(DoubleFieldProcessor.class.getName());

    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    /**
     * For olcut.
     */
    private DoubleFieldProcessor() {}

    /**
     * Constructs a field processor which extracts a single double valued feature from the specified field name.
     * @param fieldName The field name to read.
     */
    public DoubleFieldProcessor(String fieldName) {
        this.fieldName = fieldName;
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public List<ColumnarFeature> process(String value) {
        try {
            double parsedValue = Double.parseDouble(value);
            return Collections.singletonList(new ColumnarFeature(fieldName, "value", parsedValue));
        } catch (NumberFormatException ex) {
            if (!value.trim().isEmpty()) {
                logger.warning(String.format("Non-double value %s in %s", value, fieldName));
            }
            return Collections.emptyList();
        }

    }

    @Override
    public GeneratedFeatureType getFeatureType() {
        return GeneratedFeatureType.REAL;
    }

    @Override
    public DoubleFieldProcessor copy(String newFieldName) {
        return new DoubleFieldProcessor(newFieldName);
    }

    @Override
    public String toString() {
        return "DoubleFieldProcessor(fieldName=" + getFieldName() + ")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldProcessor");
    }
}
