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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Tokenizer;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

/**
 * This class was originally written for the purpose of document indexing in an
 * information retrieval context (principally used in Sun Labs' Minion search
 * engine). It was refactored here to implement the Tokenizer interface taking
 * care that the the 'ngram' tokens had correct character offsets. This is
 * typically not required in the document indexing context - but is essential
 * in other kinds of text processing / NLP tasks.
 * <p>
 * This tokenizer has some specific behavior in how it handles "ngram"
 * characters - i.e., those characters for which {@link #isNgram(char)} returns
 * true (CJK characters and others). For these characters, it will generate
 * tokens corresponding to character bigrams in addition to tokens corresponding
 * to token unigrams. Most of the other tokenizers will generate tokens that
 * have no overlapping spans but here the character bigram tokens will overlap
 * with the character unigram tokens.
 * <p>
 * This tokenizer uses bigram tokenization whenever it encounters 'ngram'
 * characters in the CJK range (among others see {@link #isNgram(char)}). It
 * otherwise tokenizes using punctuation and whitespace separators to separate
 * words. Within runs of 'ngram' characters the tokenizer will generate tokens
 * corresponding to two adjacent characters in addition to tokens corresponding
 * to each character. The tokens corresponding to character bigrams may overlap
 * with the previous and next token. An end-of-line between two 'ngram'
 * characters is ignored (i.e., a character bigram token will be created.)
 * <p>
 * For example, a sequence of three Chinese characters, 非常感, would tokenize as
 * three WORD type tokens: 非, 常, and 感 and two NGRAM type tokens: 非常 and 常感.
 * Here these tokens will have character offsets that correspond to the
 * character offsets into the text. Here are the tokens listed with their
 * character offsets:
 * <ul>
 * <li>非[0,1]</li>
 * <li>非常[0,2]</li>
 * <li>常[1,2]</li>
 * <li>常感[1,3]</li>
 * <li>感[2,3]</li>
 * </ul>
 */
public class UniversalTokenizer implements Tokenizer {

    /**
     * The length of the longest token that we will generate.
     */
    protected int maxTokenLength = 256;
    private boolean eofReached = false;

    /**
     * The character position in the character sequence that we're tokenizing.
     */
    private int pos;

    /**
     * The starting offset of the current buffer in the token stream.
     */
    private int start;

    /**
     * If <code>true</code> then unigrams will be generated for each n-gram
     * character.
     */
    private boolean generateUnigrams = true;

    /**
     * If <code>true</code> then character bigrams will be generated for each n-gram
     * character as defined by {@link #isNgram(char)}.
     */
    private boolean generateNgrams = true;
    /**
     * The state of the tokenizer determined by previous history.
     */
    private State state;
    /**
     * The character sequence that we're currently processing.
     */
    private CharSequence cs;
    /**
     * The token that we're building.
     */
    private char[] buffer;
    /**
     * A string representation of the current token.
     */
    private String currToken;
    /**
     * The current type of the token.
     */
    private Token.TokenType currType;
    /**
     * The current word position of the token.
     */
    private int currPos;
    /**
     * The starting offset of the current token.
     */
    private int startOffset;
    /**
     * The ending offset of the current token.
     */
    private int endOffset;
    /**
     * The length of the current token we're building.
     */
    private int tokenLength;
    /**
     * Whether this is the first token.
     */
    private boolean firstToken;
    /**
     * Is the tokenizer ready?
     */
    private boolean ready;
    @Config(description="Send punctuation through as tokens.")
    private boolean sendPunct = false;
    /**
     * A set of tokens that were generated and need to be returned.
     */
    private Queue<Range> queuedTokens;
    private Queue<Range> pool;
    /**
     * The current character that we're processing.
     */
    private char c;

    /**
     * Constructs a universal tokenizer.
     * @param sendPunct if sendPunct is true, then the tokenizer will generate punctuation tokens.
     */
    public UniversalTokenizer(boolean sendPunct) {
        super();
        this.sendPunct = sendPunct;
        this.buffer = new char[maxTokenLength];
        this.tokenLength = 0;
        this.state = State.SKIPPING;
        this.queuedTokens = new LinkedList<>();
        this.pool = new LinkedList<>();
    }

    /**
     * Constructs a universal tokenizer which doesn't send punctuation.
     */
    public UniversalTokenizer() {
        this(false);
    }

    /**
     * A quick check for whether a character should be kept in a word or should
     * be removed from the word if it occurs at one of the ends. An
     * approximation of Character.isLetterOrDigit, but is faster and more
     * correct, since it doesn't count the smart quotes as letters.
     *
     * @param c The character to check.
     * @return True if the input is a letter or digit.
     */
    public static boolean isLetterOrDigit(char c) {
        if ((c <= 122 && c >= 97)
                || // most frequent: lowercase a...z
                (c <= 90 && c >= 65)
                || // frequent: uppercase A...Z
                (c <= 57 && c >= 48) // frequent: numbers 0...9
        ) {
            return true;
        } else if ((c <= 96)
                || // includes whitespace
                (c == 210 || c == 211)
                || // (smart quotes)
                (c >= 123 && c <= 127) // {|}~DEL
        ) {
            return false;
        } else if ((c >= 3021 && c <= 3029)
                || // Hangzhou-style numerals
                (c >= 65 && c <= 90)
                || // frequent: uppercase A...Z
                (c >= 48 && c <= 57) // frequent: numbers 0...9
        ) {
            return true;
        } else {
            return Character.isLetterOrDigit(c);
        }
    }

    /**
     * A quick check for whether a character is a digit.
     *
     * @param c The character to check
     * @return True if the input is a digit.
     */
    public static boolean isDigit(char c) {
        if ((c <= 57 && c >= 48) // most frequent: ASCII numbers 0...9
        ) {
            return true;
        } else if (c <= 255) {
            return false;
        } else {
            return Character.isDigit(c);
        }
    }

    /**
     * A quick check for whether a character is whitespace.
     *
     * @param c The character to check
     * @return True if the input is a whitespace character.
     */
    public static boolean isWhitespace(char c) {
        //test for white space
        if ((c == 32)
                || // Space
                (c <= 13 && c >= 9)
                || // Tab, Linefeed, PageUp, Page, Return
                (c <= 4 && c >= 1) // STX, SOT, ETX (Enter), EOT
        ) {
            return true;
        } else if (c <= 255) {
            return false;
        } else {
            return Character.isWhitespace(c);
        }
    }

    /**
     * A quick check for a character in a language that may not separate words
     * with whitespace (includes Arabic, CJK, and Thai). Uses Unicode Standard
     * Version 2.0.
     *
     * @param c The character to check
     * @return True if the input character is in a region which is not whitespace separated.
     */
    public static boolean isNgram(char c) {
        // Test for characters that may not separate words with white
        // space and therefore require bigram treatment.
        // Uses Unicode Standard Version 2.0.
        if (c > '\u3002' && c <= '\uD7FF') {           // (CJK Characters)
            return (c < '\u3040' || c > '\u30FF');   // - Hiragana and Katakana
        } else if ((c >= '\u0600' && c <= '\u06FF') || // (Arabic)
                (c >= '\uF900' && c <= '\uFAFF') || // (CJK Compatibility Ideographs)
                (c >= '\u1100' && c <= '\u11FF') || // (Hangul Jamo)
                (c >= '\uFB50' && c <= '\uFE2F') || // (Arabic Presentation Forms-A)
                (c >= '\uFE30' && c <= '\uFE4F') || // (CJK Compatibility Forms)
                (c >= '\uFE70' && c <= '\uFEFF') || // (Arabic Presentation Forms-B)
                (c >= '\uFF60' && c <= '\uFFDF') || // (CJK Half Width Forms)
                (c >= '\u0E00' && c <= '\u0E7F') || // (Thai)
                (c >= '\u0E80' && c <= '\u0EFF') || // (Lao)
                (c >= '\u0F00' && c <= '\u0FBF') || // (Tibetan)
                (c >= '\u0B80' && c <= '\u0BFF') || // (Tamil)
                (c >= '\u0C00' && c <= '\u0C7F') || // (Telugu)
                (c >= '\u0C80' && c <= '\u0CFF') || // (Kannada)
                (c >= '\u0D00' && c <= '\u0D7F') || // (Malayalam)
                (c >= '\u10A0' && c <= '\u10FF')) { // (Georgian)
            return true;
        } else {
            return false;
        }
    }

    /**
     * Does this tokenizer generate unigrams?
     * @return True if the tokenizer generates unigram tokens.
     */
    public boolean isGenerateUnigrams() {
        return generateUnigrams;
    }

    /**
     * Controls if the tokenizer generates unigrams.
     * @param generateUnigrams If true generates unigram tokens.
     */
    public void setGenerateUnigrams(boolean generateUnigrams) {
        this.generateUnigrams = generateUnigrams;
    }

    /**
     * Does this tokenizer generate ngrams?
     * @return True if the tokenizer generates ngram tokens.
     */
    public boolean isGenerateNgrams() {
        return generateNgrams;
    }

    /**
     * Controls if the tokenizer generates ngrams.
     * @param generateNgrams If true generates ngram tokens.
     */
    public void setGenerateNgrams(boolean generateNgrams) {
        this.generateNgrams = generateNgrams;
    }

    /**
     * Returns the maximum token length this tokenizer will generate.
     * @return The maximum token length.
     */
    public int getMaxTokenLength() {
        return maxTokenLength;
    }

    /**
     * Sets the maximum token length this tokenizer will generate.
     * @param maxTokenLength The maximum token length.
     */
    public void setMaxTokenLength(int maxTokenLength) {
        this.maxTokenLength = maxTokenLength;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public final boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("UniversalTokenizer has not been reset.");
        }
        //
        // Do we have tokens queued up to go?
        if (queuedTokens.size() > 0) {
            handleQueued();
            return true;
        }

        //
        // If we've already read the data, then we're done.
        if (eofReached) {
            return false;
        }

        //
        // Read characters until we have one or more tokens to send.
        while (pos < cs.length()) {
            c = cs.charAt(pos);
            handleChar();
            pos++;
            if (queuedTokens.size() > 0) {
                handleQueued();
                return true;
            }
        }

        eofReached = true;
        makeTokens();
        if (queuedTokens.size() > 0) {
            handleQueued();
            return true;
        }
        return false;
    }

    private void handleQueued() {
        ready = true;
        Range range = queuedTokens.poll();
        currToken = new String(range.buff, 0, range.len);
        startOffset = range.start;
        endOffset = range.end;
        if (firstToken && range.incr == 0) {
            range.incr = 1;
            firstToken = false;
        }
        currType = range.type;
        currPos = range.incr;
        pool.offer(range);
    }

    /**
     * Handle a character to add to the token buffer.
     */
    protected void handleChar() {

        //
        // ASCII characters.
        if ((c >= 97 && c <= 122) || (c >= 65 && c <= 90)) {
            if (state == State.NGRAM) {
                makeTokens();
            }
            addChar();
            state = State.COLLECTING;
            return;
        }

        //
        // ASCII space. We need to treat other whitespace differently, depending
        // on whether we're ngram tokenizing.
        if (c == 32) {
            switch (state) {
                case COLLECTING:
                case NGRAM:
                    // The transition from collecting or n-gram to whitespace
                    // causes us to emit tokens.
                    makeTokens();
                    break;
                case SKIPPING:
                    break;
                default:
                    break;
            }
            sendPunct();
            state = State.SKIPPING;
            return;
        }

        if (isNgram(c)) {
            // CJK characters (Chinese, Japanese, Korean)
            // to be tokenized with bigram tokens.
            // (Put this test here so these languages will tokenize
            // more efficiently and it doesn't cost much for the non CJK
            // languages.)

            switch (state) {
                case SKIPPING:
                    state = State.NGRAM;
                    break;
                case COLLECTING:
                    makeTokens();
                    state = State.NGRAM;
                    break;
                case NGRAM:
                    break;
                default:
                    break;
            }
            addChar();
            return;
        }

        if (c == 0 || (state == State.NGRAM && (c >= 10 && c <= 13))) {
            // While processing ngram character regions, Linefeed, PageUp, Page, Return
            // don't do anything, so just return.
            return;
        }

        if (isWhitespace(c)) {
            // The rest of the white space characters for break:
            switch (state) {
                case COLLECTING:
                case NGRAM:
                    // The transition from collecting to whitespace
                    // causes us to emit tokens.
                    makeTokens();
                    break;
                case SKIPPING:
                    break;
                default:
                    break;
            }
            sendPunct();
            state = State.SKIPPING;
            return;
        }

        if ((c >= 48 && c <= 57) || (c > 255 && Character.isDigit(c))) {

            //
            // The digits.
            switch (state) {
                case SKIPPING:
                    state = State.COLLECTING;
                    break;
                case NGRAM:
                    makeTokens();
                    state = State.COLLECTING;
                    break;
                case COLLECTING:
                    break;
                default:
                    break;
            }
            addChar();
            return;
        }

        //
        // Any other letter or digit.
        if (isLetterOrDigit(c)) {
            if (state == State.NGRAM) {
                makeTokens();
            }
            addChar();
            state = State.COLLECTING;
            return;
        }

        // Anything other than the above cases, we break.
        if (state != State.SKIPPING) {
            makeTokens();
        }
        sendPunct();
        state = State.SKIPPING;
    }

    private void sendPunct() {
        if (sendPunct && !isWhitespace(c)) {
            Range r = getRange();
            r.punct(c, pos);
            queuedTokens.add(r);
        }
    }

    /**
     * Add a character to the buffer that we're building for a token.
     */
    protected void addChar() {

        //
        // First see if token buffer needs to be expanded.
        // Note: tokLen points to the next unused slot in token.
        if (buffer.length <= tokenLength) {
            buffer = Arrays.copyOf(buffer, tokenLength + 32);
        }

        if (tokenLength == 0) {
            start = pos;
        }
        buffer[tokenLength++] = c;

        if (tokenLength >= maxTokenLength) {
            makeTokens();
        }
    }

    @Override
    public int getStart() {
        if (ready) {
            return startOffset;
        } else {
            throw new IllegalStateException("UniversalTokenizer is not ready.");
        }
    }

    @Override
    public int getEnd() {
        if (ready) {
            return endOffset;
        } else {
            throw new IllegalStateException("UniversalTokenizer is not ready.");
        }
    }

    @Override
    public String getText() {
        if (ready) {
            return currToken;
        } else {
            throw new IllegalStateException("UniversalTokenizer is not ready.");
        }
    }

    @Override
    public Token.TokenType getType() {
        if (ready) {
            return currType;
        } else {
            throw new IllegalStateException("UniversalTokenizer is not ready.");
        }
    }

    /**
     * Gets the current position in the input.
     * @return The current position.
     */
    public int getPos() {
        return currPos;
    }

    @Override
    public Tokenizer clone() {
        try {
            UniversalTokenizer copy = (UniversalTokenizer) super.clone();

            copy.buffer = new char[maxTokenLength];
            copy.tokenLength = 0;
            copy.state = State.SKIPPING;
            copy.pool = new LinkedList<>();
            copy.queuedTokens = new LinkedList<>();
            copy.currToken = null;
            copy.ready = false;
            copy.cs = null;

            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("UniversalTokenizer is Cloneable, but clone call failed");
        }
    }

    /**
     * Reset state of tokenizer to clean slate.
     */
    @Override
    public void reset(CharSequence cs) {
        this.cs = cs;
        pos = 0;
        tokenLength = 0;
        start = -1;
        state = State.SKIPPING;
        eofReached = false;
        firstToken = true;
        c = 0;
        startOffset = -1;
        endOffset = -1;
        currToken = null;
        ready = false;
    }

    private Range getRange() {
        if (pool.isEmpty()) {
            return new Range();
        }
        return pool.remove();
    }

    /**
     * Make one or more tokens from our current collected characters.
     */
    protected void makeTokens() {

        //
        // Don't generate empty tokens.
        if (tokenLength <= 0) {
            return;
        }

        if (state == State.NGRAM) {
            // if we only have one character, then just generate a single
            // token and be done.
            if (tokenLength == 1) {
                Range range = getRange();
                range.set(buffer[0], start);
                queuedTokens.add(range);
                tokenLength = 0;
                return;
            }

            for (int i = 0; i < tokenLength; i++) {
                if (generateUnigrams) {
                    // Generate a unigram for this character.
                    Range range = getRange();
                    range.set(buffer[i], start + i);
                    queuedTokens.add(range);
                }
                if (generateNgrams && i < tokenLength - 1) {
                    // Generate a bigram for this character.
                    Range range = getRange();
                    range.set(buffer[i], buffer[i + 1], start + i);
                    queuedTokens.add(range);
                }
            }
        } else {
            // Generate one token from the buffer.
            Range range = getRange();
            range.set(buffer, tokenLength, start);
            queuedTokens.add(range);
        }
        tokenLength = 0;
    }

    private enum State {
        SKIPPING,
        COLLECTING,
        NGRAM,
    }

}
