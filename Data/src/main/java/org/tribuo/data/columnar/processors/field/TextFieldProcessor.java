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
import org.tribuo.Feature;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.text.TextPipeline;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A {@link FieldProcessor} which takes a text field and runs a {@link TextPipeline} on it
 * to generate features.
 */
public class TextFieldProcessor implements FieldProcessor {

    /**
     * The name of the field that values will be drawn from.
     */
    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    @Config(mandatory = true,description="Text processing pipeline to use.")
    private TextPipeline pipeline;

    /**
     * Constructs a field processor which uses the supplied text pipeline to process
     * the field value.
     * @param fieldName The field name to read.
     * @param pipeline The text processing pipeline to use.
     */
    public TextFieldProcessor(String fieldName, TextPipeline pipeline) {
        this.fieldName = fieldName;
        this.pipeline = pipeline;
    }

    /**
     * For olcut.
     */
    private TextFieldProcessor() {}

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public List<ColumnarFeature> process(String value) {
        if ((value == null) || (value.isEmpty())) {
            return Collections.emptyList();
        } else {
            return wrapFeatures(fieldName,pipeline.process("",value));
        }
    }

    @Override
    public GeneratedFeatureType getFeatureType() {
        return GeneratedFeatureType.TEXT;
    }

    /**
     * Note: the copy shares the text pipeline with the original. This may induce multithreading issues if
     * the underlying pipeline is not thread safe. Tribuo builtin pipelines are thread safe.
     * @param newFieldName The new field name for the copy.
     * @return A copy of this TextFieldProcessor with the new field name.
     */
    @Override
    public TextFieldProcessor copy(String newFieldName) {
        return new TextFieldProcessor(newFieldName,pipeline);
    }

    /**
     * Convert the {@link Feature}s from a text pipeline into {@link ColumnarFeature}s with the right field name.
     * @param fieldName The field name to prepend.
     * @param inputFeatures The features to convert.
     * @return A list of columnar features.
     */
    public static List<ColumnarFeature> wrapFeatures(String fieldName, List<Feature> inputFeatures) {
        if (inputFeatures.isEmpty()) {
            return Collections.emptyList();
        } else {
            List<ColumnarFeature> list = new ArrayList<>();

            for (Feature f : inputFeatures) {
                ColumnarFeature newF = new ColumnarFeature(fieldName, f.getName(), f.getValue());
                list.add(newF);
            }

            return list;
        }
    }

    @Override
    public String toString() {
        return "TextFieldProcessor(fieldName=" + getFieldName() + ",textPipeline="+pipeline.toString()+")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldProcessor");
    }
}
