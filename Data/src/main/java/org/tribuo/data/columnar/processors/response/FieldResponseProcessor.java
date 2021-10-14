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

package org.tribuo.data.columnar.processors.response;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.ConfigurableName;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.data.columnar.ResponseProcessor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * A response processor that returns the value(s) in a given (set of) fields.
 * <p>
 * We support specifying field names and default values both singly through {@link #fieldName} and {@link #defaultValue}
 * and in a list through {@link #fieldNames} and {@link #defaultValues}. The constructors and configuration preprocessing
 * have differing behaviors based on which fields are populated:
 * <ul>
 *     <li> {@link #fieldNames} and {@link #defaultValues} are both populated and the same length: fieldNames[i]'s defaultValue is defaultValues[i]
 *     <li> {@link #fieldNames} and {@link #defaultValue} are both populated: defaultValue is broadcast across all fieldNames
 *     <li> {@link #fieldName} and {@link #defaultValue} are both populated: fieldNames[0] == fieldName, defaultValues[0] == defaultValue
 * </ul>
 * All other settings are invalid.
 */
public class FieldResponseProcessor<T extends Output<T>> implements ResponseProcessor<T> {

    @Config(description="The field name to read.")
    @Deprecated
    private String fieldName;

    @Config(description="Default value to return if one isn't found.")
    private String defaultValue;

    @Config(mandatory = true, description="The output factory to use.")
    private OutputFactory<T> outputFactory;

    @Config(description = "A list of field names to read, you should use only one of this or fieldName.")
    private List<String> fieldNames;

    @Config(description = "A list of default values to return if one isn't found, one for each field")
    private List<String> defaultValues;

    @Config(description = "Whether to display field names as part of the generated label, defaults to false")
    private boolean displayField = false;

    @Config(description = "Uppercase the value before converting to output.")
    private boolean uppercase = true;

    @ConfigurableName
    private String configName;

    @Override
    public void postConfig() {
        /*
         * Canonically all internal logic is driven by
         * fieldNames and defaultValues, so this method takes values populated in fieldName and defaultValue and sets them
         * appropriately.
         */
        boolean bothFieldNamesPopulated = fieldName != null && fieldNames != null;
        boolean neitherFieldNamesPopulated = fieldName == null && fieldNames == null;
        boolean multipleFieldNamesPopulated = fieldNames != null;
        boolean singleFieldNamePopulated = fieldName != null;

        boolean bothDefaultValuesPopulated = defaultValues != null && defaultValue != null;
        boolean neitherDefaultValuesPopulated = defaultValue == null && defaultValues == null;
        boolean multipleDefaultValuesPopulated = defaultValues != null;
        boolean singleDefaultValuePopulated = defaultValue != null;

        if (bothFieldNamesPopulated || neitherFieldNamesPopulated) {
            throw new PropertyException(configName, "fieldName, FieldNames", "exactly one of fieldName or fieldNames must be populated");
        } else if (bothDefaultValuesPopulated || neitherDefaultValuesPopulated) {
            throw new PropertyException(configName, "defaultValue, defaultValues", "exactly one of defaultValue or defaultValues must be populated");
        } else if(multipleFieldNamesPopulated && multipleDefaultValuesPopulated && fieldNames.size() != defaultValues.size()) { //sizes don't match
            throw new PropertyException(configName, "defaultValues", "must match the length of fieldNames");
        } else if(multipleFieldNamesPopulated && singleDefaultValuePopulated) {
            defaultValues = Collections.nCopies(fieldNames.size(), defaultValue);
            defaultValue = null;
        } else if(singleFieldNamePopulated && multipleDefaultValuesPopulated) {
            throw new PropertyException(configName, "defaultValues", "if fieldName is populated, defaultValues must be blank");
        } else if(singleFieldNamePopulated && singleDefaultValuePopulated) {
            fieldNames = Collections.singletonList(fieldName);
            fieldName = null;
            defaultValues = Collections.singletonList(defaultValue);
            defaultValue = null;
        } // the case where both defaultValues and fieldNames are populated and their sizes match requires no action
    }

    /**
     * For olcut.
     */
    private FieldResponseProcessor() {}

    /**
     * Constructs a response processor which passes the field value through the
     * output factory.
     * Uppercases the value before generating the output.
     * @param fieldName The field to read.
     * @param defaultValue The default value to extract if it's not found.
     * @param outputFactory The output factory to use.
     */
    public FieldResponseProcessor(String fieldName, String defaultValue, OutputFactory<T> outputFactory) {
        this(Collections.singletonList(fieldName), defaultValue, outputFactory);
    }

    /**
     * Constructs a response processor which passes the field value through the
     * output factory.
     * Uppercases the value before generating the output.
     * @param fieldNames The fields to read.
     * @param defaultValue The default value to extract if it's not found.
     * @param outputFactory The output factory to use.
     */
    public FieldResponseProcessor(List<String> fieldNames, String defaultValue, OutputFactory<T> outputFactory) {
        this(fieldNames, Collections.nCopies(fieldNames.size(), defaultValue), outputFactory);
    }

    /**
     * Constructs a response processor which passes the field value through the
     * output factory. fieldNames and defaultValues must be the same length.
     * Uppercases the value before generating the output.
     * @param fieldNames The field to read.
     * @param defaultValues The default value to extract if it's not found.
     * @param outputFactory The output factory to use.
     */
    public FieldResponseProcessor(List<String> fieldNames, List<String> defaultValues, OutputFactory<T> outputFactory) {
        this(fieldNames, defaultValues, outputFactory, false);
    }

    /**
     * Constructs a response processor which passes the field value through the
     * output factory. fieldNames and defaultValues must be the same length.
     * Uppercases the value before generating the output.
     * @param fieldNames The field to read.
     * @param defaultValues The default value to extract if it's not found.
     * @param outputFactory The output factory to use.
     * @param displayField whether to include field names in the generated labels.
     */
    public FieldResponseProcessor(List<String> fieldNames, List<String> defaultValues, OutputFactory<T> outputFactory, boolean displayField) {
        this(fieldNames,defaultValues,outputFactory,displayField,true);
    }

    /**
     * Constructs a response processor which passes the field value through the
     * output factory. fieldNames and defaultValues must be the same length.
     * @param fieldNames The field to read.
     * @param defaultValues The default value to extract if it's not found.
     * @param outputFactory The output factory to use.
     * @param displayField Whether to include field names in the generated output.
     * @param uppercase Whether to uppercase the value before generating the output.
     */
    public FieldResponseProcessor(List<String> fieldNames, List<String> defaultValues, OutputFactory<T> outputFactory, boolean displayField, boolean uppercase) {
        if(fieldNames.size() != defaultValues.size()) {
            throw new IllegalArgumentException("fieldNames and defaultValues must be the same length");
        }
        this.fieldNames = fieldNames;
        this.defaultValues = defaultValues;
        this.outputFactory = outputFactory;
        this.displayField = displayField;
        this.uppercase = uppercase;
    }

    @Deprecated
    @Override
    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    @Deprecated
    public String getFieldName() {
        return fieldNames.get(0);
    }

    @Deprecated
    @Override
    public Optional<T> process(String value) {
        return process(Collections.singletonList(value));
    }

    @Override
    public Optional<T> process(List<String> values) {
        if(values.size() != fieldNames.size()) {
            throw new IllegalArgumentException("values must have the same length as fieldNames. Got values: " + values.size() + " fieldNames: "  + fieldNames.size());
        }
        List<String> responses = new ArrayList<>();
        String prefix = "";
        for(int i=0; i < fieldNames.size(); i++) {
            if (displayField) {
                prefix = fieldNames.get(i) + "=";
            }
            String val;
            if (uppercase) {
                val = values.get(i).toUpperCase().trim();
            } else {
                val = values.get(i).trim();
            }
            val = val.isEmpty() ? defaultValues.get(i) : val;
            responses.add(prefix + val);
        }
        return Optional.of(outputFactory.generateOutput(fieldNames.size() == 1 ? responses.get(0) : responses));
    }

    @Override
    public List<String> getFieldNames() {
        return fieldNames;
    }

    @Override
    public String toString() {
        return "FieldResponseProcessor(fieldNames="+ fieldNames.toString() + ",displayField="+displayField+",uppercase="+uppercase+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
