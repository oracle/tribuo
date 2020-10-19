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

/**
 * A {@link FieldProcessor} which converts the field name and value into a feature with a value of {@link IdentityProcessor#FEATURE_VALUE}.
 */
public class IdentityProcessor implements FieldProcessor {

    /**
     * The value of the emitted features.
     */
    public static final double FEATURE_VALUE = 1.0;

    /**
     * The name of the field that values will be drawn from.
     */
    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    /**
     * Constructs a field processor which emits a single feature with a specific value
     * and uses the field name and field value as the feature name.
     * @param fieldName The field name to read.
     */
    public IdentityProcessor(String fieldName) {
        this.fieldName = fieldName;
    }

    /**
     * For olcut.
     */
    private IdentityProcessor() {}

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public List<ColumnarFeature> process(String value) {
        return Collections.singletonList(new ColumnarFeature(fieldName,value,FEATURE_VALUE));
    }

    @Override
    public GeneratedFeatureType getFeatureType() {
        return GeneratedFeatureType.BINARISED_CATEGORICAL;
    }

    @Override
    public IdentityProcessor copy(String newFieldName) {
        return new IdentityProcessor(newFieldName);
    }

    @Override
    public String toString() {
        return "IdentityProcessor(fieldName=" + getFieldName() + ")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldProcessor");
    }
}
