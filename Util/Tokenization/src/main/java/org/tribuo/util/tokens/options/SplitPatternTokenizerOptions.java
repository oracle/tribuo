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
import org.tribuo.util.tokens.impl.SplitPatternTokenizer;

/**
 * CLI options for a {@link SplitPatternTokenizer}.
 */
public class SplitPatternTokenizerOptions implements TokenizerOptions {

    @Option(longName = "sp-tokenizer-regex", usage = "A regex that defines the splits (not the tokens)")
    private String splitPatternRegex = SplitPatternTokenizer.SIMPLE_DEFAULT_PATTERN;

    @Override
    public Tokenizer getTokenizer() {
        return new SplitPatternTokenizer(splitPatternRegex);
    }
}
