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

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Tokenizer;

/**
 * A convenience class for when you are required to provide a tokenizer but you
 * don't actually want to split up the text into tokens.  This tokenizer will
 * serve up a single "token" corresponding to the input text.
 */
public class NonTokenizer implements Tokenizer {

    private CharSequence cs;

    private boolean done = false;

    /**
     * Constructs a NonTokenizer.
     */
    public NonTokenizer() { }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public void reset(CharSequence cs) {
        this.cs = cs;
        this.done = false;
    }

    @Override
    public boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("NonTokenizer has not been reset.");
        }
        if (!done) {
            done = true;
            return true;
        }
        return false;
    }

    @Override
    public String getText() {
        if (done) {
            return cs.toString();
        } else {
            throw new IllegalStateException("NonTokenizer isn't ready.");
        }
    }

    @Override
    public int getStart() {
        if (done) {
            return 0;
        } else {
            throw new IllegalStateException("NonTokenizer isn't ready.");
        }
    }

    @Override
    public int getEnd() {
        if (done) {
            return cs.length();
        } else {
            throw new IllegalStateException("NonTokenizer isn't ready.");
        }
    }

    @Override
    public Token.TokenType getType() {
        if (done) {
            return Token.TokenType.WORD;
        } else {
            throw new IllegalStateException("NonTokenizer isn't ready.");
        }
    }

    @Override
    public NonTokenizer clone() {
        try {
            NonTokenizer copy = (NonTokenizer) super.clone();
            copy.done = false;
            copy.cs = null;
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new Error("Assertion error, NonTokenizer is Cloneable.");
        }
    }

}
