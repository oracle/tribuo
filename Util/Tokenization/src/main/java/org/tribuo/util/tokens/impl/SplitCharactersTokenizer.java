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
import org.tribuo.util.tokens.Token.TokenType;
import org.tribuo.util.tokens.Tokenizer;

import java.util.Arrays;

/**
 * This implementation of {@link Tokenizer} is instantiated with an array of
 * characters that are considered split characters. That is, the split
 * characters define where to split the input text. It's a very simplistic
 * tokenizer that has one simple exceptional case that it handles: how to deal
 * with split characters that appear in between digits (e.g. 3/5 and 3.1415).
 * It's not really very general purpose, but may suffice for some use cases.
 * <p>
 * In addition to the split characters specified it also splits on anything
 * that is considered whitespace by {@link Character#isWhitespace(char)}.
 * @author Philip Ogren
 */
public class SplitCharactersTokenizer implements Tokenizer {

    public static final char[] DEFAULT_SPLIT_CHARACTERS = new char[]{'*', '(', ')', '&', '[', ']', '{', '}', '`',
            '\'', '|', ';', ':', '\\', '!', '-', '?'};
    public static final char[] DEFAULT_SPLIT_EXCEPTING_IN_DIGITS_CHARACTERS = new char[]{'.', ',', '/',};

    @Config(description="The characters to split on.")
    private char[] splitCharacters = DEFAULT_SPLIT_CHARACTERS;

    @Config(description="The characters to split on unless we're in a number.")
    private char[] splitXDigitsCharacters = DEFAULT_SPLIT_EXCEPTING_IN_DIGITS_CHARACTERS;

    private CharSequence cs;

    private int start;

    private int end;

    private int p;

    private StringBuilder token = new StringBuilder();

    private boolean ready;

    public SplitCharactersTokenizer() {
    }

    /**
     * @param splitCharacters        characters to be replaced with a space in the
     *                               input text (e.g. "abc|def" becomes "abc def")
     * @param splitXDigitsCharacters characters to be replaced with a space in
     *                               the input text except in the circumstance where the character immediately
     *                               adjacent to the left and right are digits (e.g. "abc.def" becomes "abc
     *                               def" but "3.1415" remains "3.1415").
     */
    public SplitCharactersTokenizer(char[] splitCharacters, char[] splitXDigitsCharacters) {
        this.splitCharacters = splitCharacters;
        this.splitXDigitsCharacters = splitXDigitsCharacters;
    }

    /**
     * Creates a tokenizer that splits on whitespace.
     * @return A whitespace tokenizer.
     */
    public static SplitCharactersTokenizer createWhitespaceTokenizer() {
        return new SplitCharactersTokenizer(new char[0], new char[0]);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public void reset(CharSequence cs) {
        this.cs = cs;
        start = -1;
        end = -1;
        p = 0;
        token.delete(0, token.length());
        ready = false;
    }

    @Override
    public boolean advance() {
        if (cs == null) {
            throw new IllegalStateException("SplitCharactersTokenizer has not been reset.");
        }
        if (p >= cs.length()) {
            return false;
        }
        token.delete(0, token.length());
        while (p < cs.length()) {
            char c = cs.charAt(p);
            //
            // First, let's figure out if this is a character that we
            // want to keep in a token. We want to keep a character if it's
            // not one of our split characters or if it's one of the "keep in
            // digits" characters and it's surrounded by digits.
            boolean keepCharacter = !(isSplitCharacter(c) || (isSplitXDigitCharacter(c) && (p == 0
                    || p == cs.length() - 1
                    || !Character.isDigit(cs.charAt(p - 1))
                    || !Character.isDigit(cs.charAt(p + 1)))));

            p++;
            //
            // If we want to keep it, then go ahead and do that and remember
            // where the end of the token is.
            if (keepCharacter) {
                //
                // If this is the first character that we're keeping, remember
                // where the token started.
                if (token.length() == 0) {
                    start = p - 1;
                }
                token.append(c);
                end = p;
            }

            //
            // OK, if we didnt want to keep this character, and we've already
            // collected some stuff, then we've got a token to send, so let's
            // break out of the loop. This should allow us to skip runs of
            // breaking characters.
            if (!keepCharacter && token.length() > 0) {
                break;
            }
        }

        //
        // We advanced if we have some stuff collected.
        if (token.length() > 0) {
            ready = true;
            return true;
        } else {
            return false;
        }
    }

    @Override
    public String getText() {
        if (ready) {
            return token.toString();
        } else {
            throw new IllegalStateException("SplitCharactersTokenizer is not ready.");
        }
    }

    @Override
    public int getStart() {
        if (ready) {
            return start;
        } else {
            throw new IllegalStateException("SplitCharactersTokenizer is not ready.");
        }
    }

    @Override
    public int getEnd() {
        if (ready) {
            return end;
        } else {
            throw new IllegalStateException("SplitCharactersTokenizer is not ready.");
        }
    }

    @Override
    public TokenType getType() {
        if (ready) {
            return TokenType.WORD;
        } else {
            throw new IllegalStateException("SplitCharactersTokenizer is not ready.");
        }
    }

    @Override
    public SplitCharactersTokenizer clone() {
        try {
            SplitCharactersTokenizer copy = (SplitCharactersTokenizer) super.clone();
            copy.token = new StringBuilder();
            copy.splitCharacters = splitCharacters == null ? null : Arrays.copyOf(splitCharacters, splitCharacters.length);
            copy.splitXDigitsCharacters = splitXDigitsCharacters == null ? null : Arrays.copyOf(splitXDigitsCharacters, splitXDigitsCharacters.length);
            copy.ready = false;
            copy.cs = null;
            return copy;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError("SplitCharactersTokenizer is Cloneable, but clone call failed");
        }
    }

    /**
     * Is this character a split character for this tokenizer instance.
     * @param c The character to check.
     * @return True if it's a split character.
     */
    public boolean isSplitCharacter(char c) {
        return isCharacter(c, splitCharacters) || Character.isWhitespace(c);
    }

    /**
     * Is this character a split character except inside a digit for this tokenizer instance.
     * @param c The character to check.
     * @return True if it's a split character.
     */
    public boolean isSplitXDigitCharacter(char c) {
        return isCharacter(c, splitXDigitsCharacters);
    }

    private boolean isCharacter(char c, char[] chars) {
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
     * @return A copy of the split characters.
     */
    public char[] getSplitCharacters() {
        return Arrays.copyOf(splitCharacters,splitCharacters.length);
    }

    /**
     * Returns a copy of the split characters except inside digits.
     * @return A copy of the split characters.
     */
    public char[] getSplitXDigitsCharacters() {
        return Arrays.copyOf(splitXDigitsCharacters,splitXDigitsCharacters.length);
    }

}
