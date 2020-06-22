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
 * Processes the response into quartiles and emits them as classification outputs.
 */
public class QuartileResponseProcessor<T extends Output<T>> implements ResponseProcessor<T> {

    @Config(mandatory = true,description="The string to emit.")
    private String name;

    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    @Config(mandatory = true,description="The quartile to use.")
    private Quartile quartile;

    @Config(mandatory = true,description="The output factory to use.")
    private OutputFactory<T> outputFactory;

    /**
     * For olcut.
     */
    private QuartileResponseProcessor() {}

    public QuartileResponseProcessor(String name, String fieldName, Quartile quartile, OutputFactory<T> outputFactory) {
        this.name = name;
        this.fieldName = fieldName;
        this.quartile = quartile;
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
        if(value == null) {
            return Optional.of(outputFactory.generateOutput(name + ":NONE"));
        }
        double dv = Double.parseDouble(value);
        T output;
        if (dv <= quartile.getLowerMedian()) {
            output = outputFactory.generateOutput(name + ":first");
        } else if (dv > quartile.getLowerMedian() && dv <= quartile.getMedian()) {
            output = outputFactory.generateOutput(name + ":second");
        } else if (dv > quartile.getMedian() && dv <= quartile.getUpperMedian()) {
            output = outputFactory.generateOutput(name + ":third");
        } else {
            output = outputFactory.generateOutput(name + ":fourth");
        }
        return Optional.of(output);
    }

    @Override
    public String toString() {
        return "QuartileResponseProcessor(fieldName="+ fieldName +",quartile="+quartile.toString()+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
