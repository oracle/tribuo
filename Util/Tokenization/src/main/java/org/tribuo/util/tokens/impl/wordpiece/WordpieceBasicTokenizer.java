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

import org.tribuo.util.tokens.impl.SplitFunctionTokenizer;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 * This is a tokenizer that is used "upstream" of {@link WordpieceTokenizer} and
 * implements much of the functionality of the '<a href=
 * "https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py#L355">BasicTokenizer</a>'
 * implementation in huggingface. One minor difference in this implementation is
 * that there is no set of "never_split" tokens used here. Those are handled by
 * {@link WordpieceTokenizer}.
 */
public class WordpieceBasicTokenizer extends SplitFunctionTokenizer {

    /**
     * Creates a {@link SplitFunction} that is used by the super class
     * {@link SplitFunctionTokenizer} to determine how and where the tokenizer
     * splits the input.
     * 
     * @param tokenizeChineseChars split Chinese characters into separate tokens?
     * @return The splitting function.
     */
    public static SplitFunction createSplitFunction(boolean tokenizeChineseChars) {

        return (codepoint, index, cs) -> {
            if (Character.isWhitespace(codepoint)) {
                return SplitResult.SPLIT_AT;
            }
            if (codepoint == 160) { // \u00a0 (NO-BREAK SPACE)
                return SplitResult.SPLIT_AT;
            }
            if (isPunctuation(codepoint)) {
                return SplitResult.SPLIT_BEFORE_AND_AFTER_PUNCTUATION;
            }
            if (tokenizeChineseChars && isChinese(codepoint)) {
                return SplitResult.SPLIT_BEFORE_AND_AFTER_WORD;
            }
            if (codepoint == 0 || codepoint == 0xFFFD || isControl(codepoint)) {
                return SplitResult.SPLIT_AT;
            }

            return SplitResult.NO_SPLIT_WORD;
        };

    }

    /**
     * Determines if the input code point should be considered a character that is punctuation.
     * This will return true for all ascii characters that are not letters or digits and for any
     * character whose Character type is defined as punctuation.  See {@link Character#getType(int)}.
     * @param codepoint The codepoint to check.
     * @return True if the codepoint is punctuation, false otherwise.
     */
    public static boolean isPunctuation(int codepoint) {
        if (codepoint >= 33 && codepoint <= 47) {
            return true;
        }
        if (codepoint >= 58 && codepoint <= 64) {
            return true;
        }
        if (codepoint >= 91 && codepoint <= 96) {
            return true;
        }
        if (codepoint >= 123 && codepoint <= 126) {
            return true;
        }

        int charType = Character.getType(codepoint);
        if (charType == Character.DASH_PUNCTUATION || charType == Character.START_PUNCTUATION
                || charType == Character.END_PUNCTUATION || charType == Character.CONNECTOR_PUNCTUATION
                || charType == Character.OTHER_PUNCTUATION || charType == Character.INITIAL_QUOTE_PUNCTUATION
                || charType == Character.FINAL_QUOTE_PUNCTUATION) {
            return true;
        }

        return false;
    }

    /**
     * Determines if the provided codepoint is a Chinese character or not.
     * @param codepoint a codepoint
     * @return True if the codepoint is a Chinese character, false otherwise.
     */
    public static boolean isChinese(int codepoint) {
        if ((codepoint >= 0x4E00 && codepoint <= 0x9FFF) || (codepoint >= 0x3400 && codepoint <= 0x4DBF)
                || (codepoint >= 0x20000 && codepoint <= 0x2A6DF) || (codepoint >= 0x2A700 && codepoint <= 0x2B73F)
                || (codepoint >= 0x2B740 && codepoint <= 0x2B81F) || (codepoint >= 0x2B820 && codepoint <= 0x2CEAF)
                || (codepoint >= 0xF900 && codepoint <= 0xFAFF) || (codepoint >= 0x2F800 && codepoint <= 0x2FA1F)) {
            return true;
        }
        return false;
    }

    /**
     * Determines if the provided codepoint is a control character or not.
     * @param codepoint The codepoint to check.
     * @return True if it's a control character, false otherwise.
     */
    public static boolean isControl(int codepoint) {
        char c = Character.toChars(codepoint)[0];
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        int charType = Character.getType(codepoint);
        if (charType == Character.CONTROL || charType == Character.FORMAT || charType == Character.PRIVATE_USE
                || charType == Character.SURROGATE) {
            return true;
        }
        return false;
    }

    @Config(description = "split on Chinese tokens?")
    private boolean tokenizeChineseChars = true;

    /**
     * Constructs a default tokenizer which tokenizes Chinese characters.
     */
    public WordpieceBasicTokenizer() {
        this.postConfig();
    }

    /**
     * Constructs a tokenizer.
     * @param tokenizeChineseChars Should the Chinese characters be split into individual tokens.
     */
    public WordpieceBasicTokenizer(boolean tokenizeChineseChars) {
        this.tokenizeChineseChars = tokenizeChineseChars;
        this.postConfig();
    }

    @Override
    public void postConfig() {
        this.splitFunction = createSplitFunction(this.tokenizeChineseChars);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public WordpieceBasicTokenizer clone() {
        return new WordpieceBasicTokenizer(this.tokenizeChineseChars);
    }
}