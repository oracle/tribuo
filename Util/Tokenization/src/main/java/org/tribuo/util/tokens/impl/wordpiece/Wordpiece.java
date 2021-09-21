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

package org.tribuo.util.tokens.impl.wordpiece;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.util.IOUtil;

/**
 * This is vanilla implementation of the Wordpiece algorithm as found here:
 * 
 * <a href=
 * "https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py">
 * https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py</a>
 * 
 * <p>
 * Please refer to the class definition for <code>WordpieceTokenizer</code>. It
 * does not include any of the tokenization work that is typically performed
 * before wordpiece is called as is done in the above-referenced implementation.
 * That functionality is provided by {@link WordpieceTokenizer} and
 * {@link WordpieceBasicTokenizer}.
 * 
 */
public class Wordpiece implements Configurable {

    /**
     * The default unknown token string.
     */
    public static final String DEFAULT_UNKNOWN_TOKEN = "[UNK]";

    @Config(mandatory=true, description="path to a vocabulary data file.")
    private String vocabPath;
    @Config(mandatory=false, description="the value to use for 'UNKNOWN' tokens. Defaults to '[UNK]' which is a common default in BERT-based solutions.")
    private String unknownToken = DEFAULT_UNKNOWN_TOKEN;
    @Config(mandatory=false, description="the maximum number of characters per word to consider. This helps eliminate doing extra work on pathological cases.")
    private int maxInputCharactersPerWord = 100;

    private Set<String> vocab;

    /**
     * For OLCUT.
     */
    @SuppressWarnings("unused")
    private Wordpiece() { }

    /**
     * Constructs a Wordpiece using the supplied vocab.
     * <p>
     * Sets the unknown token to {@link #DEFAULT_UNKNOWN_TOKEN}.
     * @param vocab The wordpiece vocabulary.
     */
    public Wordpiece(Set<String> vocab) {
        this(vocab, DEFAULT_UNKNOWN_TOKEN);
    }

    /**
     * Constructs a Wordpiece using the supplied vocabulary and unknown token.
     * @param vocab The wordpiece vocabulary.
     * @param unknownToken The unknown token.
     */
    public Wordpiece(Set<String> vocab, String unknownToken) {
        this(vocab, unknownToken, 100);
    }

    /**
     * Initializes an instance of Wordpiece with the given vocabulary, unknown
     * token, and max word length.
     * 
     * @param vocab                     the pre-trained wordpiece vocabulary. See
     *                                  the contents of e.g.,
     *                                  <a href="https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt">
     *                                  https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt</a>
     * @param unknownToken              a string used to indicate a token was not
     *                                  found in the vocabulary - typically "[UNK]"
     * @param maxInputCharactersPerWord a maximum to shield against looping over
     *                                  character-by-character pathologically long
     *                                  "tokens"
     */
    public Wordpiece(Set<String> vocab, String unknownToken, int maxInputCharactersPerWord) {
        this.vocab = Collections.unmodifiableSet(vocab);
        this.unknownToken = unknownToken;
        this.maxInputCharactersPerWord = maxInputCharactersPerWord;
    }

    /**
     * Constructs a wordpiece by reading the vocabulary from the supplied path.
     * @param vocabPath The path to the wordpiece vocabulary.
     */
    public Wordpiece(String vocabPath) {
        this.vocabPath = vocabPath;
        try {
            this.postConfig();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Initializes an instance of Wordpiece with the given vocabulary, unknown
     * token, and max word length.
     * 
     * @param vocabPath                 Path to the pre-trained wordpiece vocabulary. See
     *                                  the contents of e.g.
     *                                  https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
     * @param unknownToken              a string used to indicate a token was not
     *                                  found in the vocabulary - typically "[UNK]"
     * @param maxInputCharactersPerWord a maximum to shield against looping over
     *                                  character-by-character pathologically long
     *                                  "tokens"
     */
    public Wordpiece(String vocabPath, String unknownToken, int maxInputCharactersPerWord) {
        this.vocabPath = vocabPath;
        this.unknownToken = unknownToken;
        this.maxInputCharactersPerWord = maxInputCharactersPerWord;
        try {
            this.postConfig();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        this.vocab = Collections.unmodifiableSet(new HashSet<>(IOUtil.getLines(this.vocabPath)));
    }

    /**
     * Executes Wordpiece tokenization on the provided token. Note that tokens
     * corresponding to word suffixes as indicated in the provided vocabulary with
     * the sequence "##" prepended to the entry may be returned by this method. This
     * method does not perform whitespace tokenization or any other preprocessing.
     * This method does not lowercase the input token or otherwise modify it in any
     * way.
     * 
     * @param token the token to apply Wordpiece tokenization to.
     * @return tokens corresponding to Wordpiece tokenization applied to the input
     *         text. Some tokens may have a prefix "##" as described above. Some
     *         tokens may correspond to an unknown token as specified during
     *         initialization (default "[UNK]")
     */
    public List<String> wordpiece(String token) {
        if (token.length() > this.maxInputCharactersPerWord) {
            return Collections.singletonList(this.unknownToken);
        }

        List<String> subTokens = new ArrayList<>();

        boolean isBad = false;
        int start = 0;
        while (start < token.length()) {
            int end = token.length();
            String currentSubstring = null;
            while (start < end) {
                String substring = token.substring(start, end);
                if (start > 0) {
                    substring = "##" + substring;
                }
                if (this.vocab.contains(substring)) {
                    currentSubstring = substring;
                    break;
                }
                end--;
            }
            if (currentSubstring == null) {
                isBad = true;
                break;
            }
            subTokens.add(currentSubstring);
            start = end;
        }
        if (isBad) {
            return Collections.singletonList(this.unknownToken);
        } else {
            return subTokens;
        }
    }

    /**
     * a getter for the "unknown" token specified during initialization.
     * 
     * @return the "unknown" token name - defaults to "[UNK]"
     */
    public String getUnknownToken() {
        return unknownToken;
    }

    /**
     * a getter for the maximum character count for a token to consider when
     * {@link #wordpiece(String)} is applied to a token. This value is set at
     * initialization and defaults to 100. Token values passed to that method that
     * are not tokenized and the result of {@link #getUnknownToken()} is returned
     * instead.
     * 
     * @return the maximum length of a token that will be analyzed by
     *         {@link #wordpiece(String)}.
     */
    public int getMaxInputCharactersPerWord() {
        return maxInputCharactersPerWord;
    }

}
