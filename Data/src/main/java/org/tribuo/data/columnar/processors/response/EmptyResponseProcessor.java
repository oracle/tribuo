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

import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 *  A {@link ResponseProcessor} that always emits an empty optional.
 *  <p>
 *  This class is designed to be used when loading columnar datasets
 *  which will never have a response (e.g., for clustering or anomaly detection).
 *  <p>
 *  It still requires an output factory, even though it's never used to generate
 *  an output, because the output factory provides the type for the columnar infrastructure.
 */
public final class EmptyResponseProcessor<T extends Output<T>> implements ResponseProcessor<T> {

    /**
     * The field name this response processor looks for, which is ignored anyway as this processor always returns {@link Optional#empty()}.
     */
    public static final String FIELD_NAME = "TRIBUO##NULL_RESPONSE_PROCESSOR";

    @Config(mandatory = true,description="Output factory to type the columnar loader.")
    private OutputFactory<T> outputFactory;

    /**
     * for OLCUT.
     */
    private EmptyResponseProcessor() {}

    /**
     * Constructs a response processor which never emits a response.
     * <p>
     * It contains an output factory as this types the whole columnar infrastructure.
     * @param outputFactory The output factory to use.
     */
    public EmptyResponseProcessor(OutputFactory<T> outputFactory) {
        this.outputFactory = outputFactory;
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Deprecated
    @Override
    public String getFieldName() {
        return FIELD_NAME;
    }

    /**
     * This is a no-op as the empty response processor doesn't inspect a field.
     * @param fieldName The field name.
     */
    @Deprecated
    @Override
    public void setFieldName(String fieldName) { }

    /**
     * This method always returns {@link Optional#empty}.
     * @param value The value to process.
     * @return {@link Optional#empty}.
     */
    @Deprecated
    @Override
    public Optional<T> process(String value) {
        return Optional.empty();
    }

    /**
     * This method always returns {@link Optional#empty}.
     * @param values The values to process.
     * @return {@link Optional#empty}.
     */
    @Override
    public Optional<T> process(List<String> values) {
        return Optional.empty();
    }

    @Override
    public List<String> getFieldNames() {
        return Collections.singletonList(FIELD_NAME);
    }

    @Override
    public String toString() {
        return "EmptyResponseProcessor(outputFactory="+outputFactory.toString()+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
