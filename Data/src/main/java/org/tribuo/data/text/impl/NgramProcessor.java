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
import org.tribuo.Feature;
import org.tribuo.data.text.TextProcessingException;
import org.tribuo.data.text.TextProcessor;
import org.tribuo.util.tokens.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * A text processor that will generate token ngrams of a particular size.
 */
public class NgramProcessor implements TextProcessor {

    private static final Logger logger = Logger.getLogger(NgramProcessor.class.getName());

    @Config(description="n in the n-gram to emit.")
    private int n = 2;

    @Config(description="Value to emit for each n-gram.")
    private double value = 1.0;

    @Config(mandatory = true,description="Tokenizer to use.")
    private Tokenizer tokenizer;

    private ThreadLocal<Tokenizer> tokenizerThreadLocal;
    
    /**
     * Creates a processor that will generate token ngrams of size {@code n}.
     * 
     * @param tokenizer The tokenizer to use to process text.
     * @param n the size of the ngram to generate
     * @param value the value we will put in the new features.
     */
    public NgramProcessor(Tokenizer tokenizer, int n, double value) {
        if (n < 1) {
            throw new IllegalArgumentException("n = " + n + ", must be a positive integer.");
        }
        this.n = n;
        this.value = value;
        this.tokenizer = tokenizer;
        this.tokenizerThreadLocal = ThreadLocal.withInitial(() -> {try { return this.tokenizer.clone(); } catch (CloneNotSupportedException e) { throw new IllegalArgumentException("Tokenizer not cloneable",e); }});
    }

    /**
     * For olcut.
     */
    private NgramProcessor() {}

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        this.tokenizerThreadLocal = ThreadLocal.withInitial(() -> {try { return tokenizer.clone(); } catch (CloneNotSupportedException e) { throw new IllegalArgumentException("Tokenizer not cloneable",e); }});
    }

    @Override
    public List<Feature> process(String text) throws TextProcessingException {
        return innerProcess(n+"-N=",text);
    }

    @Override
    public List<Feature> process(String tag, String text) throws TextProcessingException {
        if (tag == null || tag.isEmpty()) {
            return innerProcess(n+"-N=",text);
        } else {
            return innerProcess(tag + "-" + n + "-N=", text);
        }
    }

    private List<Feature> innerProcess(String tag, String text) {
        List<Feature> ret = new ArrayList<>();

        List<String> words = tokenizerThreadLocal.get().split(text);

        if (words.size() < n) {
            return ret;
        }

        StringBuilder ngram = new StringBuilder();
        for (int start = 0, end = n; end <= words.size(); start++, end++) {
            ngram.delete(0,ngram.length());
            ngram.append(tag);
            for (int i = start; i < end; ++i) {
                ngram.append(words.get(i));
                ngram.append('/');
            }
            ngram.deleteCharAt(ngram.length()-1);
            if (ngram.length() > 0 && Character.isLetterOrDigit(ngram.charAt(0))) {
                String ngramString = ngram.toString();
                ret.add(new Feature(ngramString, value));
            }
        }
        return ret;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"TextProcessor");
    }

}
