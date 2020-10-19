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
 *  A {@link ResponseProcessor} that takes a single value of the
 *  field as the positive class and all other values as the negative
 *  class.
 */
public class BinaryResponseProcessor<T extends Output<T>> implements ResponseProcessor<T> {

    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    @Config(mandatory = true,description="The string which triggers a positive response.")
    private String positiveResponse;

    @Config(mandatory = true,description="Output factory to use to create the response.")
    private OutputFactory<T> outputFactory;

    @Config(description="The positive response to emit.")
    private String positiveName = "1";

    @Config(description="The negative response to emit.")
    private String negativeName = "0";

    /**
     * for OLCUT.
     */
    private BinaryResponseProcessor() {}

    /**
     * Constructs a binary response processor which emits a positive value for a single string
     * and a negative value for all other field values.
     * @param fieldName The field name to read.
     * @param positiveResponse The positive response to look for.
     * @param outputFactory The output factory to use.
     */
    public BinaryResponseProcessor(String fieldName, String positiveResponse, OutputFactory<T> outputFactory) {
        this.fieldName = fieldName;
        this.positiveResponse = positiveResponse;
        this.outputFactory = outputFactory;
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Deprecated
    @Override
    public void setFieldName(String fieldName) {
        this.fieldName = fieldName;
    }

    @Override
    public Optional<T> process(String value) {
        return Optional.of(outputFactory.generateOutput(positiveResponse.equals(value) ? positiveName : negativeName));
    }

    @Override
    public String toString() {
        return "BinaryResponseProcessor(fieldName="+ fieldName +", positiveResponse="+ positiveResponse +", positiveName="+positiveName +", negativeName="+negativeName+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
