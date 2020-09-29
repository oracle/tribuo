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

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * This implementation of {@link Tokenizer} is instantiated with a regular
 * expression pattern which determines how to split a string into tokens. That
 * is, the pattern defines the "splits", not the tokens. For example, to
 * tokenize on white space provide the pattern "\s+".
 *
 * @author Philip Ogren
 */
public class SplitPatternTokenizer implements Tokenizer {

    /**
     * The default split pattern, which is [\.,]?\s+.
     */
    public static final String SIMPLE_DEFAULT_PATTERN = "[\\.,]?\\s+";

    @Config(description="The regex to split with.")
    private String splitPatternRegex = SIMPLE_DEFAULT_PATTERN;

    private Pattern splitPattern;

    private CharSequence cs;

    private int start;

    private int end;

    private Matcher matcher;

    private int prevMatchEnd;

    private boolean done;

    private boolean ready;

    /**
     * Initializes a case insensitive tokenizer with the pattern [\.,]?\s+
     */
    public SplitPatternTokenizer() {
        postConfig();
    }

    /**
     * Constructs a splitting tokenizer using the supplied regex.
     * @param splitPatternRegex The regex to use.
     */
    public SplitPatternTokenizer(String splitPatternRegex) {
        this.splitPatternRegex = splitPatternRegex;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        splitPattern = Pattern.compile(splitPatternRegex);
        ready = false;
        cs = null;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    /**
     * Gets the String form of the regex in use.
     * @return The regex.
     */
    public String getSplitPatternRegex() {
        return splitPatternRegex;
    }

    @Override
    public void reset(CharSequence cs) {
        this.cs = cs;
        matcher = splitPattern.matcher(cs);
        start = -1;
        end = -1;
        prevMatchEnd = 0;
        done = false;
        ready = false;
    }

    @Override
    public boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("SplitPatternTokenizer has not been reset.");
        }
        //
        // We've gotten everything.
        if (done) {
            return false;
        }
        if (matcher.find()) {
            //
            // We might get a match at the start of the string, so reset and 
            // call advance to see if we can find a later match.
            if (matcher.start() == 0) {
                prevMatchEnd = matcher.end();
                return advance();
            }
            //
            // A regular match, so the actual text runs from the end of the 
            // previous match to the start of this one.
            start = prevMatchEnd;
            end = matcher.start();
            prevMatchEnd = matcher.end();
            ready = true;
        } else {
            //
            // Handle the end of the string, keeping in mind that the last match
            // might have included the end of the string already.
            start = prevMatchEnd;
            end = cs.length();
            done = true;
            ready = start < end;
        }
        return ready;
    }

    @Override
    public String getText() {
        if (ready) {
            return cs.subSequence(start, end).toString();
        } else {
            throw new IllegalStateException("SplitPatternTokenizer is not ready.");
        }
    }

    @Override
    public int getStart() {
        if (ready) {
            return start;
        } else {
            throw new IllegalStateException("SplitPatternTokenizer is not ready.");
        }
    }

    @Override
    public int getEnd() {
        if (ready) {
            return end;
        } else {
            throw new IllegalStateException("SplitPatternTokenizer is not ready.");
        }
    }

    @Override
    public Token.TokenType getType() {
        if (ready) {
            return Token.TokenType.WORD;
        } else {
            throw new IllegalStateException("SplitPatternTokenizer is not ready.");
        }
    }

    @Override
    public SplitPatternTokenizer clone() {
        try {
            SplitPatternTokenizer copy = (SplitPatternTokenizer) super.clone();
            copy.postConfig(); //ready is set in postConfig.
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("SplitPatternTokenizer is Cloneable, but the clone call failed.");
        }
    }
}
