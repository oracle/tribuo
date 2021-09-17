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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Feature;
import org.tribuo.data.text.FeatureAggregator;
import org.tribuo.data.text.TextPipeline;
import org.tribuo.data.text.TextProcessingException;
import org.tribuo.data.text.TextProcessor;
import org.tribuo.util.tokens.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An example implementation of {@link TextPipeline}. Generates unique ngrams.
 */
public class BasicPipeline implements TextPipeline {

    private static final Logger logger = Logger.getLogger(BasicPipeline.class.getName());

    private List<TextProcessor> processors = new ArrayList<>();
    private FeatureAggregator aggregator = new UniqueAggregator();

    @Config(mandatory = true,description="Tokenizer to use.")
    private Tokenizer tokenizer;

    @Config(description="n in the n-gram to emit.")
    private int ngram = 2;

    /**
     * Constructs a basic text pipeline which tokenizes the input and generates word
     * n-gram features in the range 1 to {@code ngram}.
     * @param tokenizer The tokenizer.
     * @param ngram The size of the n-grams to generate.
     */
    public BasicPipeline(Tokenizer tokenizer, int ngram) {
        this.tokenizer = tokenizer;
        this.ngram = ngram;
        postConfig();
    }

    /**
     * For OLCUT.
     */
    private BasicPipeline() {}

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (ngram < 1) {
            throw new PropertyException("","ngram","ngram must be positive, found " + ngram);
        }
        for (int i = 1; i <= ngram; ++i) {
            processors.add(new NgramProcessor(tokenizer,i,1.0));
        }
    }

    @Override
    public String toString() {
        return ngram + "gramPipeline({1.."+ngram+"}-grams)";
    }

    @Override
    public List<Feature> process(String tag, String data) {
        List<Feature> features = new ArrayList<>();

        for (TextProcessor p : processors) {
            try {
                features.addAll(p.process(tag,data));
            } catch (TextProcessingException e) {
                logger.log(Level.INFO, String.format("TextProcessingException thrown by processor %s with text %s",p,data), e);
            }
        }

        return aggregator.aggregate(features);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"TextPipeline");
    }

}
