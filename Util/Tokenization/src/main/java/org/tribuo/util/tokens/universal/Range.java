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

package org.tribuo.util.tokens.universal;

import org.tribuo.util.tokens.Token;

/**
 * A range currently being segmented.
 */
public final class Range implements CharSequence {
    /**
     * The character buffer.
     */
    public char[] buff = new char[16];
    /**
     * The token length.
     */
    public int len;
    /**
     * The start index.
     */
    public int start;
    /**
     * The end index.
     */
    public int end;
    /**
     * The value to increment by.
     */
    public int incr;
    /**
     * The current token type.
     */
    public Token.TokenType type;

    Range() {}

    /**
     * Sets the first two characters in the range, and the type to NGRAM.
     * @param c1 The first character.
     * @param c2 The second character.
     * @param start The start value.
     */
    public void set(char c1, char c2, int start) {
        buff[0] = c1;
        buff[1] = c2;
        this.start = start;
        this.end = start + 2;
        this.len = 2;
        this.incr = 0;
        this.type = Token.TokenType.NGRAM;
    }

    /**
     * Sets the first character in the range.
     * @param c The first character.
     * @param start The start value.
     */
    public void set(char c, int start) {
        buff[0] = c;
        this.start = start;
        this.end = start + 1;
        this.len = 1;
        this.incr = 1;
        this.type = Token.TokenType.WORD;
    }

    /**
     * Sets the character range.
     * @param buff The characters.
     * @param len The length of the character buffer.
     * @param start The start index.
     */
    public void set(char[] buff, int len, int start) {
        if (this.buff.length < buff.length) {
            this.buff = new char[buff.length + 1];
        }
        System.arraycopy(buff, 0, this.buff, 0, len);
        this.len = len;
        this.start = start;
        this.end = start + len;
        this.incr = 1;
        this.type = Token.TokenType.WORD;
    }

    /**
     * Sets this range to represent a punctuation character.
     * @param p The punctuation character.
     * @param start The start index.
     */
    public void punct(char p, int start) {
        buff[0] = p;
        this.len = 1;
        this.start = Math.max(start, 0);
        this.end = this.start + 1;
        this.incr = 0;
        this.type = Token.TokenType.PUNCTUATION;
    }

    /**
     * Sets the token type.
     * @param type The token type.
     */
    public void setType(Token.TokenType type) {
        this.type = type;
    }

    @Override
    public int length() {
        return len;
    }

    @Override
    public char charAt(int index) {
        if (index < len) {
            return buff[index];
        }
        throw new IndexOutOfBoundsException(String.format("index %d exceeds length %d", index, len));
    }

    @Override
    public CharSequence subSequence(int start, int end) {
        Range r = new Range();
        System.arraycopy(buff, start, r.buff, 0, end - start);
        r.start = 0;
        r.len = end - start;
        r.end = r.len;
        return r;
    }

    @Override
    public String toString() {
        return new String(buff, 0, len) + " " + type + " " + start + " " + end;
    }
}
