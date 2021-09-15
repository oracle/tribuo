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

package org.tribuo.util.tokens.options;

import com.oracle.labs.mlrg.olcut.config.Option;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.impl.NonTokenizer;
import org.tribuo.util.tokens.impl.ShapeTokenizer;
import org.tribuo.util.tokens.universal.UniversalTokenizer;

import java.util.logging.Logger;

/**
 * CLI Options for all the tokenizers in the core package.
 */
public class CoreTokenizerOptions implements TokenizerOptions {

    private static final Logger logger = Logger.getLogger(CoreTokenizerOptions.class.getName());
    /**
     * Options for the break iterator tokenizer.
     */
    public BreakIteratorTokenizerOptions breakIteratorOptions;
    /**
     * Options for the split characters tokenizer.
     */
    public SplitCharactersTokenizerOptions splitCharactersTokenizerOptions;
    /**
     * Options for the split pattern tokenizer.
     */
    public SplitPatternTokenizerOptions splitPatternTokenizerOptions;
    /**
     * Type of tokenizer
     */
    @Option(longName = "core-tokenizer-type", usage = "Type of tokenizer")
    public CoreTokenizerType coreTokenizerType = CoreTokenizerType.SPLIT_CHARACTERS;

    @Override
    public Tokenizer getTokenizer() {
        Tokenizer tokenizer;
        logger.info("Using " + coreTokenizerType);
        switch (coreTokenizerType) {
            case BREAK_ITERATOR:
                tokenizer = breakIteratorOptions.getTokenizer();
                break;
            case SPLIT_CHARACTERS:
                tokenizer = splitCharactersTokenizerOptions.getTokenizer();
                break;
            case NON:
                tokenizer = new NonTokenizer();
                break;
            case SHAPE:
                tokenizer = new ShapeTokenizer();
                break;
            case SPLIT_PATTERN:
                tokenizer = splitPatternTokenizerOptions.getTokenizer();
                break;
            case UNIVERSAL:
                tokenizer = new UniversalTokenizer();
                break;
            default:
                throw new IllegalArgumentException("Unknown tokenizer " + coreTokenizerType);
        }
        return tokenizer;
    }

    /**
     * Tokenizer type.
     */
    public enum CoreTokenizerType {
        /**
         * Creates a {@link org.tribuo.util.tokens.impl.BreakIteratorTokenizer}.
         */
        BREAK_ITERATOR,
        /**
         * Creates a {@link org.tribuo.util.tokens.impl.SplitCharactersTokenizer}.
         */
        SPLIT_CHARACTERS,
        /**
         * Creates a {@link NonTokenizer}.
         */
        NON,
        /**
         * Creates a {@link ShapeTokenizer}.
         */
        SHAPE,
        /**
         * Creates a {@link org.tribuo.util.tokens.impl.SplitPatternTokenizer}.
         */
        SPLIT_PATTERN,
        /**
         * Creates a {@link UniversalTokenizer}.
         */
        UNIVERSAL
    }

}
