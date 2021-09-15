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

package org.tribuo.util.tokens.impl;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Tokenizer;

import java.text.BreakIterator;
import java.util.Locale;

/**
 * A tokenizer wrapping a {@link BreakIterator} instance.
 */
public class BreakIteratorTokenizer implements Tokenizer {

    @Config(mandatory = true, description="The locale language tag string.")
    private String localeStr;

    private Locale locale;

    private BreakIterator breakIterator;

    private CharSequence cs;

    private int start;

    private int startOffset;
    private int endOffset;

    private String token;

    private boolean ready;

    /**
     * Default constructor for configuration system.
     */
    @SuppressWarnings("unused")
    private BreakIteratorTokenizer() {}

    /**
     * Constructs a BreakIteratorTokenizer using the specified locale.
     * @param locale The locale to use.
     */
    public BreakIteratorTokenizer(Locale locale) {
        this.locale = locale;
        this.localeStr = locale.toLanguageTag();
        breakIterator = BreakIterator.getWordInstance(locale);
        ready = false;
        cs = null;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        locale = Locale.forLanguageTag(localeStr);
        breakIterator = BreakIterator.getWordInstance(locale);
        ready = false;
        cs = null;
    }

    /**
     * Returns the locale string this tokenizer uses.
     * @return The locale string.
     */
    public String getLanguageTag() {
        return localeStr;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public void reset(CharSequence cs) {
        this.cs = cs;
        breakIterator.setText(cs.toString());
        start = breakIterator.first();
        startOffset = -1;
        endOffset = -1;
        token = null;
        ready = false;
    }

    @Override
    public boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("BreakIteratorTokenizer has not been reset.");
        }
        int end = breakIterator.next();
        while (end != BreakIterator.DONE) {
            token = cs.subSequence(start, end).toString();
            startOffset = start;
            endOffset = end;
            start = end;
            if (!token.trim().isEmpty()) {
                ready = true;
                return true;
            } else {
                end = breakIterator.next();
            }
        }

        return false;
    }

    @Override
    public String getText() {
        if (ready) {
            return token;
        } else {
            throw new IllegalStateException("BreakIteratorTokenizer is not ready.");
        }
    }

    @Override
    public int getStart() {
        if (ready) {
            return startOffset;
        } else {
            throw new IllegalStateException("BreakIteratorTokenizer is not ready.");
        }
    }

    @Override
    public int getEnd() {
        if (ready) {
            return endOffset;
        } else {
            throw new IllegalStateException("BreakIteratorTokenizer is not ready.");
        }
    }

    @Override
    public Token.TokenType getType() {
        if (ready) {
            return Token.TokenType.WORD;
        } else {
            throw new IllegalStateException("BreakIteratorTokenizer is not ready.");
        }
    }

    @Override
    public BreakIteratorTokenizer clone() {
        try {
            BreakIteratorTokenizer copy = (BreakIteratorTokenizer) super.clone();
            copy.postConfig();
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("BreakIteratorTokenizer is Cloneable, but clone call failed");
        }
    }
}

