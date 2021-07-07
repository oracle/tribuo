/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Token.TokenType;
import org.tribuo.util.tokens.Tokenizer;

/**
 * This class supports character-by-character (that is, codepoint-by-codepoint)
 * iteration over input text to create tokens. Extensions of this class are
 * initialized with a {@link SplitFunction} which will be called for each character and
 * a {@link SplitResult} consisting of a {@link SplitType} and a {@link TokenType} will be returned.
 * Tokenization is achieved based on the {@link SplitResult} returned for each
 * character. Please see notes below for each {@link SplitType} and {@link SplitResult}.
 */
public abstract class SplitFunctionTokenizer implements Tokenizer {

    /**
     * Defines different ways that a tokenizer can split the input text at a given character.
     */
    public enum SplitType {
        /**
         * the current character is added to the in-progress token (i.e. do not split on
         * the current character)
         */
        NO_SPLIT,
        /**
         * The current character will cause the in-progress token to be completed. the
         * current character will not be included in any returned token and the token
         * type of the corresponding SplitResult is ignored (See {@link SplitResult#SPLIT_AT}).
         * This SplitType may be useful for whitespace.
         */
        SPLIT_AT,
        /**
         * The current character will cause the in-progress token to be completed the
         * current character will be included in the next token. The token type of the
         * corresponding SplitResult is ignored (See {@link SplitResult#SPLIT_BEFORE}). This
         * SplitType may be useful for e.g. capitalized letters when CamelCase splitting
         * of digits when separating out a currency symbol.
         */
        SPLIT_BEFORE,
        /**
         * The current character will cause the in-progress token to be completed after
         * the current character is appended to the in-progress token. The token type of
         * the created token (that includes the current character) will be assigned the
         * type included with the {@link SplitResult}.
         */
        SPLIT_AFTER,
        /**
         * The current character should cause the in-progress token to be completed. The
         * token assigned to the in-progress token will be whatever was previously
         * assigned to the previous character. This token will be followed by a second
         * single-character token consisting of the current character. The token type
         * assigned to this second token will be provided with the {@link SplitResult}.
         */
        SPLIT_BEFORE_AND_AFTER
    }

    /**
     * A combination of a {@link SplitType} and a {@link TokenType}. The TokenType of some
     * SplitResult values are ignored and so not every combination of SplitType and
     * TokenType is provided. For example, {@link SplitType#SPLIT_AT} and
     * {@link SplitType#SPLIT_BEFORE} (as described above) create tokens whose types have
     * already been determined.
     */
    public enum SplitResult {
        /**
         * Not a split, is a word.
         */
        NO_SPLIT_WORD(SplitType.NO_SPLIT, TokenType.WORD),
        /**
         * Not a split, is a ngram.
         */
        NO_SPLIT_NGRAM(SplitType.NO_SPLIT, TokenType.NGRAM),
        /**
         * Not a split, is punctuation.
         */
        NO_SPLIT_PUNCTUATION(SplitType.NO_SPLIT, TokenType.PUNCTUATION),
        /**
         * Not a split, is whitespace.
         */
        NO_SPLIT_WHITESPACE(SplitType.NO_SPLIT, TokenType.WHITESPACE),
        /**
         * Not a split, is a prefix.
         */
        NO_SPLIT_PREFIX(SplitType.NO_SPLIT, TokenType.PREFIX),
        /**
         * Not a split, is a suffix.
         */
        NO_SPLIT_SUFFIX(SplitType.NO_SPLIT, TokenType.SUFFIX),
        /**
         * Not a split, is infix.
         */
        NO_SPLIT_INFIX(SplitType.NO_SPLIT, TokenType.INFIX),
        /**
         * Not a split, is unknown.
         */
        NO_SPLIT_UNKNOWN(SplitType.NO_SPLIT, TokenType.UNKNOWN),
        /**
         * Split at.
         */
        SPLIT_AT(SplitType.SPLIT_AT, TokenType.WORD), //the token type is ignored
        /**
         * Split before.
         */
        SPLIT_BEFORE(SplitType.SPLIT_BEFORE, TokenType.WORD), //the token type is ignored
        /**
         * Split after a word.
         */
        SPLIT_AFTER_WORD(SplitType.SPLIT_AFTER, TokenType.WORD),
        /**
         * Split after a ngram.
         */
        SPLIT_AFTER_NGRAM(SplitType.SPLIT_AFTER, TokenType.NGRAM),
        /**
         * Split after punctuation.
         */
        SPLIT_AFTER_PUNCTUATION(SplitType.SPLIT_AFTER, TokenType.PUNCTUATION),
        /**
         * Split after whitespace.
         */
        SPLIT_AFTER_WHITESPACE(SplitType.SPLIT_AFTER, TokenType.WHITESPACE),
        /**
         * Split after a prefix.
         */
        SPLIT_AFTER_PREFIX(SplitType.SPLIT_AFTER, TokenType.PREFIX),
        /**
         * Split after a suffix.
         */
        SPLIT_AFTER_SUFFIX(SplitType.SPLIT_AFTER, TokenType.SUFFIX),
        /**
         * Split after infix.
         */
        SPLIT_AFTER_INFIX(SplitType.SPLIT_AFTER, TokenType.INFIX),
        /**
         * Split after an unknown value.
         */
        SPLIT_AFTER_UNKNOWN(SplitType.SPLIT_AFTER, TokenType.UNKNOWN),
        /**
         * Split before and after a word.
         */
        SPLIT_BEFORE_AND_AFTER_WORD(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.WORD),
        /**
         * Split before and after a ngram.
         */
        SPLIT_BEFORE_AND_AFTER_NGRAM(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.NGRAM),
        /**
         * Split before and after punctuation.
         */
        SPLIT_BEFORE_AND_AFTER_PUNCTUATION(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.PUNCTUATION),
        /**
         * Split before and after whitespace.
         */
        SPLIT_BEFORE_AND_AFTER_WHITESPACE(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.WHITESPACE),
        /**
         * Split before and after prefix.
         */
        SPLIT_BEFORE_AND_AFTER_PREFIX(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.PREFIX),
        /**
         * Split before and after suffix.
         */
        SPLIT_BEFORE_AND_AFTER_SUFFIX(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.SUFFIX),
        /**
         * Split before and after infix.
         */
        SPLIT_BEFORE_AND_AFTER_INFIX(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.INFIX),
        /**
         * Split before and after unknown.
         */
        SPLIT_BEFORE_AND_AFTER_UNKNOWN(SplitType.SPLIT_BEFORE_AND_AFTER, TokenType.UNKNOWN);

        /**
         * The split type.
         */
        public final SplitType splitType;
        /**
         * The token type.
         */
        public final TokenType tokenType;

        SplitResult(SplitType splitType, TokenType tokenType) {
            this.splitType = splitType;
            this.tokenType = tokenType;
        }
    }

    /**
     * An interface for checking if the text should be split at the supplied codepoint.
     */
    @FunctionalInterface
    public static interface SplitFunction {
        /**
         * Applies the split function.
         * @param codepoint The codepoint to check.
         * @param index The character index.
         * @param cs The sequence that's being split.
         * @return How the sequence should be split.
         */
        public SplitResult apply(int codepoint, int index, CharSequence cs);
    }

    protected SplitFunction splitFunction;

    /**
     * Constructs a tokenizer, used by OLCUT.
     */
    protected SplitFunctionTokenizer() { }

    /**
     * Creates a new tokenizer using the supplied split function.
     * @param splitFunction The split function.
     */
    public SplitFunctionTokenizer(SplitFunction splitFunction) {
        super();
        this.splitFunction = splitFunction;
    }

    private String cs;

    private int start;

    private int p;

    private StringBuilder tokenSb = new StringBuilder();

    private TokenType currentType = TokenType.WORD;

    private Token currentToken;

    private Token nextToken;

    private boolean ready;

    @Override
    public void reset(CharSequence cs) {
        this.cs = cs.toString();
        start = 0;
        p = 0;
        tokenSb.delete(0, tokenSb.length());
        ready = false;
    }

    @Override
    public boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("SplitFunctionTokenizer has not been reset.");
        }
        if (nextToken != null) {
            currentToken = nextToken;
            nextToken = null;
            return true;
        }
        if (p >= cs.length()) {
            return false;
        }

        currentToken = null;

        SplitResult splitResult;
        SplitType splitType;
        TokenType tokenType;

        tokenSb.delete(0, tokenSb.length());
        while (p < cs.length()) {
            int codepoint = cs.codePointAt(p);
            splitResult = splitFunction.apply(codepoint, p, cs);
            splitType = splitResult.splitType;
            tokenType = splitResult.tokenType;
            // If we want to keep it, then go ahead and do that and remember
            // where the end of the token is.
            if (splitType == SplitType.NO_SPLIT) {
                if (tokenSb.length() == 0) {
                    start = p;
                }
                p += Character.charCount(codepoint);
                tokenSb.appendCodePoint(codepoint);
                currentType = tokenType;
                continue;
            }

            if (splitType == SplitType.SPLIT_AT) {
                if (tokenSb.length() > 0) {
                    currentToken = new Token(tokenSb.toString(), start, p, currentType);
                }
                p += Character.charCount(codepoint);
                start = p;
                tokenSb.delete(0, tokenSb.length());
            } else if (splitType == SplitType.SPLIT_BEFORE) {
                if (tokenSb.length() > 0) {
                    currentToken = new Token(tokenSb.toString(), start, p, currentType);
                }
                start = p;
                tokenSb.delete(0, tokenSb.length());
                tokenSb.appendCodePoint(codepoint);
                p += Character.charCount(codepoint);
            } else if (splitType == SplitType.SPLIT_AFTER) {
                p += Character.charCount(codepoint);
                tokenSb.appendCodePoint(codepoint);
                // no need to check the length since we just added a code point
                currentToken = new Token(tokenSb.toString(), start, p, tokenType);
                tokenSb.delete(0, tokenSb.length());
                start = p;
            } else if (splitType == SplitType.SPLIT_BEFORE_AND_AFTER) {
                // wrap up the token we are currently building and then create
                // the next token which consists of just the character
                if (tokenSb.length() > 0) {
                    currentToken = new Token(tokenSb.toString(), start, p, currentType);
                    tokenSb.delete(0, tokenSb.length());
                    start = p;
                    p += Character.charCount(codepoint);
                    tokenSb.appendCodePoint(codepoint);
                    nextToken = new Token(tokenSb.toString(), start, p, tokenType);
                    tokenSb.delete(0, tokenSb.length());
                } else {
                    start = p;
                    p += Character.charCount(codepoint);
                    tokenSb.appendCodePoint(codepoint);
                    currentToken = new Token(tokenSb.toString(), start, p, tokenType);
                    tokenSb.delete(0, tokenSb.length());
                }
            }
            if (currentToken != null) {
                break;
            }
        }

        if (currentToken == null) {
            if (tokenSb.length() > 0) {
                currentToken = new Token(tokenSb.toString(), start, p, currentType);
            }
        }

        // We advanced if we have some stuff collected.
        if (currentToken != null) {
            ready = true;
            return true;
        } else {
            return false;
        }
    }

    @Override
    public String getText() {
        if (ready) {
            return currentToken.text;
        } else {
            throw new IllegalStateException("SplitFunctionTokenizer is not ready.");
        }
    }

    @Override
    public int getStart() {
        if (ready) {
            return currentToken.start;
        } else {
            throw new IllegalStateException("SplitFunctionTokenizer is not ready.");
        }
    }

    @Override
    public int getEnd() {
        if (ready) {
            return currentToken.end;
        } else {
            throw new IllegalStateException("SplitFunctionTokenizer is not ready.");
        }
    }

    @Override
    public TokenType getType() {
        return currentToken.type;
    }

    @Override
    public Tokenizer clone() throws CloneNotSupportedException {
        throw new UnsupportedOperationException(
                "abstract class SplitFunctionTokenizer does not implement clone method.  Subclasses must implement this method.");
    }

}
