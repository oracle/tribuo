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
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.data.columnar.ResponseProcessor;

import java.util.Optional;

/**
 * A response processor that returns the value in a given field.
 */
public class FieldResponseProcessor<T extends Output<T>> implements ResponseProcessor<T> {

    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    @Config(mandatory = true,description="Default value to return if one isn't found.")
    private String defaultValue;

    @Config(mandatory = true,description="The output factory to use.")
    private OutputFactory<T> outputFactory;

    /**
     * For olcut.
     */
    private FieldResponseProcessor() {}

    public FieldResponseProcessor(String fieldName, String defaultValue, OutputFactory<T> outputFactory) {
        this.fieldName = fieldName;
        this.defaultValue = defaultValue;
        this.outputFactory = outputFactory;
    }

    @Override
    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public Optional<T> process(String value) {
        String val = value == null ? defaultValue : value;
        if (val != null) {
            val = val.toUpperCase().trim();
            if (val.isEmpty()) {
                return Optional.empty();
            } else{
                return Optional.of(outputFactory.generateOutput(val));
            }
        } else {
            return Optional.empty();
        }
    }

    @Override
    public String toString() {
        return "FieldResponseProcessor(fieldName="+ fieldName +")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
