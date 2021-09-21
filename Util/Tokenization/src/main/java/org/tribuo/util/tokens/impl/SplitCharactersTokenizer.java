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

import java.util.Arrays;

import org.tribuo.util.tokens.Tokenizer;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 * This implementation of {@link Tokenizer} is instantiated with an array of
 * characters that are considered split characters. That is, the split
 * characters define where to split the input text. It's a very simplistic
 * tokenizer that has one simple exceptional case that it handles: how to deal
 * with split characters that appear in between digits (e.g., 3/5 and 3.1415).
 * It's not really very general purpose, but may suffice for some use cases.
 * <p>
 * In addition to the split characters specified it also splits on anything that
 * is considered whitespace by {@link Character#isWhitespace(char)}.
 * 
 * @author Philip Ogren
 */
public class SplitCharactersTokenizer extends SplitFunctionTokenizer {

    /**
     * The default split characters.
     */
    public static final char[] DEFAULT_SPLIT_CHARACTERS = new char[] { '*', '(', ')', '&', '[', ']', '{', '}', '`',
            '\'', '|', ';', ':', '\\', '!', '-', '?' };
    /**
     * The default characters which don't cause splits inside digits.
     */
    public static final char[] DEFAULT_SPLIT_EXCEPTING_IN_DIGITS_CHARACTERS = new char[] { '.', ',', '/', };

    /**
     * Splits tokens at the supplied characters.
     */
    public static class SplitCharactersSplitterFunction implements SplitFunction {

        private final char[] splitCharacters;

        private final char[] splitXDigitsCharacters;

        /**
         * Constructs a splitting function using the supplied split characters.
         * @param splitCharacters The characters to split on.
         * @param splitXDigitsCharacters Characters that are valid split points outside of a run of digits.
         */
        public SplitCharactersSplitterFunction(char[] splitCharacters, char[] splitXDigitsCharacters) {
            this.splitCharacters = splitCharacters;
            this.splitXDigitsCharacters = splitXDigitsCharacters;
        }

        @Override
        public SplitResult apply(int codepoint, int index, CharSequence cs) {
            if (isSplitCharacter((char) codepoint)) {
                return SplitResult.SPLIT_AT;
            }
            if (isSplitXDigitCharacter((char) codepoint)) {
                if (index == 0 || index == cs.length() - 1 || !Character.isDigit(cs.charAt(index - 1))
                        || !Character.isDigit(cs.charAt(index + 1))) {
                    return SplitResult.SPLIT_AT;
                }
            }
            return SplitResult.NO_SPLIT_WORD;
        }

        /**
         * Checks if this is a valid split character or whitespace.
         * @param c The character to check.
         * @return True if the character should split the token.
         */
        public boolean isSplitCharacter(char c) {
            return isCharacter(c, splitCharacters) || Character.isWhitespace(c);
        }

        /**
         * Checks if this a valid split character outside of a run of digits.
         * @param c The character to check.
         * @return True if the character should split the token.
         */
        public boolean isSplitXDigitCharacter(char c) {
            return isCharacter(c, splitXDigitsCharacters);
        }

    }

    @Config(description = "The characters to split on.")
    private char[] splitCharacters = DEFAULT_SPLIT_CHARACTERS;

    @Config(description = "The characters to split on unless we're in a number.")
    private char[] splitXDigitsCharacters = DEFAULT_SPLIT_EXCEPTING_IN_DIGITS_CHARACTERS;

    /**
     * Creates a default split characters tokenizer using
     * {@link #DEFAULT_SPLIT_CHARACTERS} and
     * {@link #DEFAULT_SPLIT_EXCEPTING_IN_DIGITS_CHARACTERS}.
     */
    public SplitCharactersTokenizer() {
        this.postConfig(); // I feel like I need to call this explicitly in case someone uses the default
                           // constructor
    }

    @Override
    public void postConfig() {
        this.splitFunction = new SplitCharactersSplitterFunction(splitCharacters, splitXDigitsCharacters);
    }

    /**
     * @param splitCharacters        characters to be replaced with a space in the
     *                               input text (e.g., "abc|def" becomes "abc def")
     * @param splitXDigitsCharacters characters to be replaced with a space in the
     *                               input text except in the circumstance where the
     *                               character immediately adjacent to the left and
     *                               right are digits (e.g., "abc.def" becomes "abc
     *                               def" but "3.1415" remains "3.1415").
     */
    public SplitCharactersTokenizer(char[] splitCharacters, char[] splitXDigitsCharacters) {
        this.splitCharacters = splitCharacters;
        this.splitXDigitsCharacters = splitXDigitsCharacters;
        this.postConfig();
    }

    /**
     * Creates a tokenizer that splits on whitespace.
     * 
     * @return A whitespace tokenizer.
     */
    public static SplitCharactersTokenizer createWhitespaceTokenizer() {
        return new SplitCharactersTokenizer(new char[0], new char[0]);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    /**
     * Is this character a split character for this tokenizer instance.
     * 
     * @param c The character to check.
     * @return True if it's a split character.
     */
    @Deprecated
    public boolean isSplitCharacter(char c) {
        return isCharacter(c, splitCharacters) || Character.isWhitespace(c);
    }

    /**
     * Is this character a split character except inside a digit for this tokenizer
     * instance.
     * 
     * @param c The character to check.
     * @return True if it's a split character.
     */
    @Deprecated
    public boolean isSplitXDigitCharacter(char c) {
        return isCharacter(c, splitXDigitsCharacters);
    }

    private static boolean isCharacter(char c, char[] chars) {
        if (chars == null) {
            return false;
        }
        for (char ch : chars) {
            if (ch == c) {
                return true;
            }
        }
        return false;
    }

    /**
     * Returns a copy of the split characters.
     * 
     * @return A copy of the split characters.
     */
    @Deprecated
    public char[] getSplitCharacters() {
        return Arrays.copyOf(splitCharacters, splitCharacters.length);
    }

    /**
     * Returns a copy of the split characters except inside digits.
     * 
     * @return A copy of the split characters.
     */
    @Deprecated
    public char[] getSplitXDigitsCharacters() {
        return Arrays.copyOf(splitXDigitsCharacters, splitXDigitsCharacters.length);
    }

    @Override
    public SplitCharactersTokenizer clone() {
        return new SplitCharactersTokenizer(splitCharacters, splitXDigitsCharacters);
    }

}
