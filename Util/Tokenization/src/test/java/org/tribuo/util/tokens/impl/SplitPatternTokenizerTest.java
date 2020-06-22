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

import org.tribuo.util.tokens.TokenizerTestBase;
import org.junit.jupiter.api.Test;

public class SplitPatternTokenizerTest extends TokenizerTestBase {

    @Test
    public void testTokenizer() {
        SplitPatternTokenizer tokenizer = new SplitPatternTokenizer();
        test(tokenizer, "1.0n", "1.0n");
        test(tokenizer, "1. 0n", "1", "0n");
        test(tokenizer, "a .10n", "a", ".10n");
        test(tokenizer, "a ,10n", "a", ",10n");
        test(tokenizer, "a, b, and c", "a", "b", "and", "c");

        tokenizer = new SplitPatternTokenizer("\\s+");
        test(tokenizer, "a   b c", "a", "b", "c");
        test(tokenizer, "");
        test(tokenizer, "    ");
        test(tokenizer, "    a", "a");
        test(tokenizer, "hello there!", "hello", "there!");
    }

    @Test
    public void testClone() {
        SplitPatternTokenizer tokenizer = new SplitPatternTokenizer();
        testClones(tokenizer, "1.0n", "1.0n");
        testClones(tokenizer, "1. 0n", "1", "0n");
        testClones(tokenizer, "a .10n", "a", ".10n");
        testClones(tokenizer, "a ,10n", "a", ",10n");
        testClones(tokenizer, "a, b, and c", "a", "b", "and", "c");

        tokenizer = new SplitPatternTokenizer("\\s+");
        testClones(tokenizer, "a   b c", "a", "b", "c");
        testClones(tokenizer, "");
        testClones(tokenizer, "    ");
        testClones(tokenizer, "    a", "a");
        testClones(tokenizer, "hello there!", "hello", "there!");
    }

    @Test
    public void testLdaVocab() {
        SplitPatternTokenizer tokenizer = new SplitPatternTokenizer("[0-9\\W]+");
        test(tokenizer, "asdf=fdsa", "asdf", "fdsa");
        test(tokenizer, "as\"df", "as", "df");
        test(tokenizer, "2020-2020");
        test(tokenizer, "asdf-fdsa", "asdf", "fdsa");

        tokenizer = new SplitPatternTokenizer("[^0-9A-Za-z]+");
        test(tokenizer, "asdf=fdsa", "asdf", "fdsa");
        test(tokenizer, "as\"df", "as", "df");
        test(tokenizer, "2020-2020", "2020", "2020");
        test(tokenizer, "asdf-fdsa", "asdf", "fdsa");
        test(tokenizer, "Assessment of myocardial viability in .... Rest-4-hour-24-hour 201Tl tomography.",
                "Assessment", "of", "myocardial", "viability", "in", "Rest", "4", "hour", "24", "hour", "201Tl", "tomography");
    }
}
