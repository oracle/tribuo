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
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.util.Locale;

/**
 * CLI options for a {@link BreakIteratorTokenizer}.
 */
public class BreakIteratorTokenizerOptions implements TokenizerOptions {

    /**
     * BreakIteratorTokenizer - The language tag of the locale to be used.
     */
    @Option(longName = "bi-tokenizer-language-tag", usage = "BreakIteratorTokenizer - The language tag of the locale to be used.")
    public String languageTag;

    @Override
    public Tokenizer getTokenizer() {
        return new BreakIteratorTokenizer(Locale.forLanguageTag(languageTag));
    }
}
