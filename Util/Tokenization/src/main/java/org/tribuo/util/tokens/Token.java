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

package org.tribuo.util.tokens;

/**
 * A single token extracted from a String.
 * <p>
 * Tokens are immutable.
 */
public class Token {

    public final String text;
    public final int start;
    public final int end;
    public final TokenType type;

    /**
     * Constructs a token.
     * @param text  should be equivalent to the substring of the original
     *              tokenized text for the given character offsets start and end
     * @param start the starting offset of the token
     * @param end   the ending offset of the token (exclusive or inclusive?)
     */
    public Token(String text, int start, int end) {
        this(text, start, end, TokenType.WORD);
    }

    /**
     * Constructs a token.
     * @param text  should be equivalent to the substring of the original
     *              tokenized text for the given character offsets start and end
     * @param start the starting offset of the token
     * @param end   the ending offset of the token (exclusive or inclusive?)
     * @param type  the type of the token
     */
    public Token(String text, int start, int end, TokenType type) {
        this.text = text;
        this.start = start;
        this.end = end;
        this.type = type;
    }

    /**
     * The number of characters in this token.
     * @return The number of characters.
     */
    public int length() {
        return this.end - this.start;
    }

    @Override
    public String toString() {
        return this.text + "[type=" + this.type + "," + this.start + "," + this.end + "]";
    }

    /**
     * Tokenizers may product multiple kinds of tokens, depending on the
     * application to which they're being put. For example, when processing a
     * document for highlighting during querying, we need to send through
     * whitespace and punctuation so that the document looks as it did in it's
     * original form. For most tokenizer applications, they will only send word
     * tokens.
     */
    public enum TokenType {
        WORD,
        NGRAM,
        PUNCTUATION,
        WHITESPACE
    }

}
