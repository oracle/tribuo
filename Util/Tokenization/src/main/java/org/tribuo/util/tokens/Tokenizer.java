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

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;

/**
 * An interface for things that tokenize text: breaking it into words according
 * to some set of rules.
 * <p>
 * Note that tokenizers are not guaranteed to be thread safe! Using the same
 * tokenizer from multiple threads may result in strange behavior.
 * <p>
 * Tokenizers which are not ready throw {@link IllegalStateException}
 * when {@link Tokenizer#advance} or any get method is called.
 * <p>
 * Most Tokenizers are Cloneable, and implement the Cloneable interface.
 */
public interface Tokenizer extends Configurable, Cloneable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Creates a supplier from the specified tokenizer by cloning it.
     * @param tokenizer The tokenizer to copy.
     * @return A supplier of tokenizers.
     */
    static Supplier<Tokenizer> createSupplier(Tokenizer tokenizer) {
        Supplier<Tokenizer> supplier = () -> {
            try {
                return tokenizer.clone();
            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        };
        return supplier;
    }

    /**
     * Creates a thread local source of tokenizers by making a Tokenizer supplier using {@link #createSupplier(Tokenizer)}.
     * @param tokenizer The tokenizer to copy.
     * @return A thread local for tokenizers.
     */
    static ThreadLocal<Tokenizer> createThreadLocal(Tokenizer tokenizer) {
        return ThreadLocal.withInitial(createSupplier(tokenizer));
    }

    /**
     * Resets the tokenizer so that it operates on a new sequence of characters.
     *
     * @param cs a character sequence to tokenize
     */
    public void reset(CharSequence cs);

    /**
     * Advances the tokenizer to the next token.
     *
     * @return {@code true} if there is such a token, {@code false}
     * otherwise.
     */
    public boolean advance();

    /**
     * Gets the text of the current token, as a string
     *
     * @return the text of the current token
     */
    public String getText();

    /**
     * Gets the starting character offset of the current token in the character
     * sequence
     *
     * @return the starting character offset of the token
     */
    public int getStart();

    /**
     * Gets the ending offset (exclusive) of the current token in the character
     * sequence
     *
     * @return the exclusive ending character offset for the current token.
     */
    public int getEnd();

    /**
     * Gets the type of the current token.
     *
     * @return the type of the current token.
     */
    public Token.TokenType getType();

    /**
     * Clones a tokenizer with it's configuration. Cloned tokenizers are
     * not processing the same text as the original tokenizer and need to be reset
     * with a fresh CharSequence.
     *
     * @return A tokenizer with the same configuration, but independent state.
     * @throws CloneNotSupportedException if the tokenizer isn't cloneable.
     */
    public Tokenizer clone() throws CloneNotSupportedException;

    /**
     * Generates a Token object from the current state of the tokenizer.
     * @return The token object from the current state.
     */
    default public Token getToken() {
        return new Token(getText(), getStart(), getEnd(), getType());
    }

    /**
     * Uses this tokenizer to tokenize a string and return the list of tokens
     * that were generated. Many applications will simply want to take a
     * character sequence and get a list of tokens, so this will do that for
     * them.
     *
     * <p>
     * Here is the contract of the tokenize function:
     * <ul>
     * <li>all returned tokens correspond to substrings of the input text</li>
     * <li>the tokens do not overlap</li>
     * <li>the tokens are returned in the order that they appear in the text
     * </li>
     * <li>the value of Token.text should be the same as calling
     * text.substring(token.start, token.end)
     * </ul>
     *
     * @param cs a sequence of characters to tokenize
     * @return the tokens discovered in the character sequence, in order
     * (true?).
     */
    default List<Token> tokenize(CharSequence cs) {
        if (cs == null || cs.length() == 0) {
            return Collections.emptyList();
        }
        List<Token> tokens = new ArrayList<>();
        reset(cs);
        while (advance()) {
            tokens.add(getToken());
        }
        return tokens;
    }

    /**
     * Uses this tokenizer to split a string into it's component substrings.
     * Many applications will simply want the component strings making up a
     * larger character sequence.
     *
     * @param cs the character sequence to tokenize
     * @return a list of strings making up the character sequence.
     */
    default List<String> split(CharSequence cs) {
        if (cs == null || cs.length() == 0) {
            return Collections.emptyList();
        }
        List<String> tokens = new ArrayList<>();
        reset(cs);
        while (advance()) {
            tokens.add(getText());
        }
        return tokens;
    }
}
