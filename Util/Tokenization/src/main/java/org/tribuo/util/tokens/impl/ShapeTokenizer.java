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
 * This tokenizer is loosely based on the notion of word shape which is a common
 * feature used in NLP. The idea here is that continuous runs of letters in the
 * same character class will be grouped together. White space characters are
 * used as delimiters. The character classes are: uppercase letters, lowercase
 * letters, digits, and everything else goes into its own character class. So,
 * for example, "1234abcd" would be split into "1234" and "abcd". And "!@#$"
 * would result in four tokens. Please see unit tests.
 * <p>
 * Strings are split according to whitespace and contiguous runs of characters
 * in the same character classes. Except for one exception - if uppercase
 * letters are immediately followed by lowercase letters, then we keep them
 * together. This has the effect of recognizing camel case and splits
 * "CamelCase" into "Camel" and "Case". It also splits "ABCdef AAbb" into
 * "ABCdef" and "AAbb".
 */
public class ShapeTokenizer implements Tokenizer {

    private String cs;

    private int pos;

    private String token;

    private StringBuilder tb = new StringBuilder();

    private int start;

    private int end;

    private char currClass;

    private int prevClass;

    private boolean ready;

    /**
     * Constructs a ShapeTokenizer.
     */
    public ShapeTokenizer() { }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public void reset(CharSequence cs) {
        this.cs = cs.toString();
        pos = 0;
        start = -1;
        end = -1;
        prevClass = -1;
        token = null;
        ready = false;
    }

    private char getClass(int cp) {
        if (Character.isUpperCase(cp)) {
            return 'A';
        } else if (Character.isLowerCase(cp)) {
            return 'a';
        } else if (Character.isDigit(cp)) {
            return '1';
        } else if (Character.isWhitespace(cp)) {
            return ' ';
        } else {
            return (char) cp;
        }
    }

    @Override
    public boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("ShapeTokenizer has not been reset.");
        }
        tb.delete(0, tb.length());
        start = pos;
        while (pos < cs.length()) {
            int cp = cs.codePointAt(pos);
            int lcp = Character.charCount(cp);

            currClass = getClass(cp);

            //
            // Skip spaces at the start of the token.
            if (tb.length() == 0 && currClass == ' ') {
                pos += lcp;
                start = pos;
                prevClass = currClass;
                continue;
            }

            //
            // When do we want to end the current token? When we cross a boundary
            // between token classes when we're not at the start of the string,
            // except when that boundary is between 
            // upper and lower case characters.
            if (currClass != prevClass && prevClass != -1) {
                if (!(prevClass == 'A' && currClass == 'a')) {
                    if (tb.length() > 0) {
                        token = tb.toString();
                        prevClass = currClass;
                        //
                        // Note that we're not increasing pos here: we want
                        // to work on this current character the next time that
                        // we get called!
                        ready = true;
                        return true;
                    }
                }
            }

            //
            // We didn't end the token, so collect the current character,
            // unless it's a space!
            if (currClass != ' ') {
                tb.appendCodePoint(cp);
            }
            prevClass = currClass;
            pos += lcp;
            end = pos;
        }

        if (tb.length() > 0) {
            token = tb.toString();
            ready = true;
            return true;
        }

        return false;
    }

    @Override
    public String getText() {
        if (ready) {
            return token;
        } else {
            throw new IllegalStateException("ShapeTokenizer is not ready.");
        }
    }

    @Override
    public int getStart() {
        if (ready) {
            return start;
        } else {
            throw new IllegalStateException("ShapeTokenizer is not ready.");
        }
    }

    @Override
    public int getEnd() {
        if (ready) {
            return end;
        } else {
            throw new IllegalStateException("ShapeTokenizer is not ready.");
        }
    }

    @Override
    public Token.TokenType getType() {
        if (ready) {
            return Token.TokenType.WORD;
        } else {
            throw new IllegalStateException("ShapeTokenizer is not ready.");
        }
    }

    @Override
    public ShapeTokenizer clone() {
        try {
            ShapeTokenizer copy = (ShapeTokenizer) super.clone();
            copy.tb = new StringBuilder();
            copy.ready = false;
            copy.cs = null;
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("ShapeTokenizer is Cloneable, but clone call failed");
        }
    }

}
