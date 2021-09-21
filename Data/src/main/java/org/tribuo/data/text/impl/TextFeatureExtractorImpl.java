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

package org.tribuo.data.text.impl;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Output;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.TextPipeline;
import org.tribuo.impl.ArrayExample;

import java.util.List;

/**
 * An implementation of {@link TextFeatureExtractor} that takes a
 * {@link TextPipeline} and generates {@link ArrayExample}.
 */
public class TextFeatureExtractorImpl<T extends Output<T>> implements TextFeatureExtractor<T> {

    @Config(mandatory=true,description="The text processing pipeline.")
    private TextPipeline pipeline;

    /**
     * for olcut
     */
    private TextFeatureExtractorImpl() {}

    /**
     * Constructs a text feature extractor wrapping the supplied text pipeline.
     * @param pipeline The text processing pipeline.
     */
    public TextFeatureExtractorImpl(TextPipeline pipeline) {
        this.pipeline = pipeline;
    }

    @Override
    public String toString() {
        return "TextFeatureExtractorImpl(pipeline="+pipeline.toString()+")";
    }

    @Override
    public Example<T> extract(T label, String data) {
        ArrayExample<T> example = new ArrayExample<>(label);
        List<Feature> features = pipeline.process("",data);

        example.addAll(features);

        return example;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"TextFeatureExtractor");
    }
}
