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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Processes the response into quartiles and emits them as classification outputs.
 * <p>
 * The emitted outputs for each field are of the form:
 * {@code {<fieldName>:first, <fieldName>:second, <fieldName>:third, <fieldName>:fourth} }.
 */
public class QuartileResponseProcessor<T extends Output<T>> implements ResponseProcessor<T> {

    /**
     * @deprecated This field causes issues with multidimensional outputs.
     * When populated the emitted outputs are of the form:
     * {@code {<name>:first, <name>:second, <name>:third, <name>:fourth} }.
     */
    @Config(mandatory = true,description="The string to emit.")
    @Deprecated
    private String name;

    @Config(description="The field name to read.")
    @Deprecated
    private String fieldName;

    @Config(description="The quartile to use.")
    private Quartile quartile;

    @Config(mandatory = true,description="The output factory to use.")
    private OutputFactory<T> outputFactory;

    @Config(description = "A list of field names to read, you should use only one of this or fieldName.")
    private List<String> fieldNames;

    @Config(description = "A list of quartiles to use, should have the same length as fieldNames")
    private List<Quartile> quartiles;

    @ConfigurableName
    private String configName;

    @Override
    public void postConfig() throws PropertyException, IOException {
        if (fieldName != null && fieldNames != null) {
            throw new PropertyException(configName, "fieldName, FieldNames", "only one of fieldName or fieldNames can be populated");
        } else if (fieldNames != null) {
            if(quartile != null) {
                quartiles = quartiles == null ? Collections.nCopies(fieldNames.size(), quartile) : quartiles;
            } else {
                throw new PropertyException(configName, "quartile, quartiles", "one of quartile or quartiles must be populated");
            }
            if(quartiles.size() != fieldNames.size()) {
                throw new PropertyException(configName, "quartiles", "must either be empty or match the length of fieldNames");
            }
        } else if (fieldName != null) {
            if (quartiles != null) {
                throw new PropertyException(configName, "quartiles", "if fieldName is populated, quartiles must be blank");
            }
            fieldNames = Collections.singletonList(fieldName);
            if(quartile != null) {
                quartiles = Collections.singletonList(quartile);
            } else {
                throw new PropertyException(configName, "quartile", "if fieldName is populated, quartile must be populated");
            }
        } else {
            throw new PropertyException(configName, "fieldName, fieldNames", "One of fieldName or fieldNames must be populated");
        }
    }

    /**
     * For olcut.
     */
    private QuartileResponseProcessor() {}

    /**
     * Constructs a response processor which emits 4 distinct bins for the output factory to process.
     * <p>
     * This works best with classification outputs as the discrete binning is tricky to do in other output
     * types.
     * @param name The output string to emit.
     * @param fieldName The field to read.
     * @param quartile The quartile range to use.
     * @param outputFactory The output factory to use.
     */
    public QuartileResponseProcessor(String name, String fieldName, Quartile quartile, OutputFactory<T> outputFactory) {
        this(Collections.singletonList(fieldName), Collections.singletonList(quartile), outputFactory);
        this.name = name;
    }

    /**
     * Constructs a response processor which emits 4 distinct bins for the output factory to process.
     * <p>
     * This works best with classification outputs as the discrete binning is tricky to do in other output
     * types.
     * @param fieldNames The field to read.
     * @param quartiles The quartile range to use.
     * @param outputFactory The output factory to use.
     */
    public QuartileResponseProcessor(List<String> fieldNames, List<Quartile> quartiles, OutputFactory<T> outputFactory) {
        this.fieldNames = fieldNames;
        this.quartiles = quartiles;
        this.outputFactory = outputFactory;
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
    public Optional<T> process(List<String> values) {
        if(values.size() != fieldNames.size()) {
            throw new IllegalArgumentException("values must have the same length as fieldNames. Got values: " + values.size() + " fieldNames: "  + fieldNames.size());
        }
        List<String> response = new ArrayList<>();
        for(int i=0; i< fieldNames.size(); i++) {
            String value = values.get(i);
            String prefix = name == null || name.isEmpty() ? fieldNames.get(i) : name;
            Quartile q = quartiles.get(i);
            if(value == null) {
                response.add(prefix + ":NONE");
            } else {
                double dv = Double.parseDouble(value);
                if (dv <= q.getLowerMedian()) {
                    response.add(prefix + ":first");
                } else if (dv > q.getLowerMedian() && dv <= q.getMedian()) {
                    response.add(prefix + ":second");
                } else if (dv > q.getMedian() && dv <= q.getUpperMedian()) {
                    response.add(prefix + ":third");
                } else {
                    response.add(prefix + ":fourth");
                }
            }
        }
        return Optional.of(outputFactory.generateOutput(response.size() == 1 ? response.get(0) : response));
    }

    @Override
    public List<String> getFieldNames() {
        return fieldNames;
    }

    @Override
    public String toString() {
        return "QuartileResponseProcessor(fieldNames="+ fieldNames.toString() +",quartiles=" + quartiles.toString() + ")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"ResponseProcessor");
    }
}
