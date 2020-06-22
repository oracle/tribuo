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

import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.tribuo.util.tokens.universal.UniversalTokenizer;

public class UniversalTokenizerTest extends TokenizerTestBase {

    @Test
    public void testTokenizer() {
        Tokenizer tokenizer = new UniversalTokenizer(false);
        test(tokenizer, "1.0n", "1", "0n");
        test(tokenizer, "1. 0n", "1", "0n");
        test(tokenizer, "a .10n", "a", "10n");
        test(tokenizer, "a ,10n", "a", "10n");
        test(tokenizer, "a, b, and c", "a", "b", "and", "c");
        test(tokenizer, "a   b c", "a", "b", "c");
        test(tokenizer, "");
        test(tokenizer, "    ");
        test(tokenizer, "    a", "a");
        test(tokenizer, "hello there!", "hello", "there");

        test(tokenizer, "4:36 PM", "4", "36", "PM");
        test(tokenizer, "a-b", "a", "b");
        test(tokenizer, "a*b(c)d&e[f]g{h}i`j'k|l!m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m");
        test(tokenizer, "a.b 3.1 a.4 a1.2b c3.d4 5e.6f", "a", "b", "3", "1", "a", "4", "a1", "2b", "c3", "d4", "5e", "6f");
        test(tokenizer, "a/b 3/1 a/4 a1/2b c3/d4 5e/6f", "a", "b", "3", "1", "a", "4", "a1", "2b", "c3", "d4", "5e", "6f");
        test(tokenizer, "a,b 3,1 a,4 a1,2b c3,d4 5e,6f", "a", "b", "3", "1", "a", "4", "a1", "2b", "c3", "d4", "5e", "6f");
    }

    @Test
    public void testClones() {
        Tokenizer tokenizer = new UniversalTokenizer(false);
        testClones(tokenizer, "1.0n", "1", "0n");
        testClones(tokenizer, "1. 0n", "1", "0n");
        testClones(tokenizer, "a .10n", "a", "10n");
        testClones(tokenizer, "a ,10n", "a", "10n");
        testClones(tokenizer, "a, b, and c", "a", "b", "and", "c");
        testClones(tokenizer, "a   b c", "a", "b", "c");
        testClones(tokenizer, "");
        testClones(tokenizer, "    ");
        testClones(tokenizer, "    a", "a");
        testClones(tokenizer, "hello there!", "hello", "there");

        testClones(tokenizer, "4:36 PM", "4", "36", "PM");
        testClones(tokenizer, "a-b", "a", "b");
        testClones(tokenizer, "a*b(c)d&e[f]g{h}i`j'k|l!m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m");
        testClones(tokenizer, "a.b 3.1 a.4 a1.2b c3.d4 5e.6f", "a", "b", "3", "1", "a", "4", "a1", "2b", "c3", "d4", "5e", "6f");
        testClones(tokenizer, "a/b 3/1 a/4 a1/2b c3/d4 5e/6f", "a", "b", "3", "1", "a", "4", "a1", "2b", "c3", "d4", "5e", "6f");
        testClones(tokenizer, "a,b 3,1 a,4 a1,2b c3,d4 5e,6f", "a", "b", "3", "1", "a", "4", "a1", "2b", "c3", "d4", "5e", "6f");
    }

    @Test
    public void testMore() {
        Tokenizer tokenizer = new UniversalTokenizer(true);
        test(tokenizer, "1.0n", "1", ".", "0n");
        test(tokenizer, "hello there!", "hello", "there", "!");
        test(tokenizer, "4:36 PM", "4", ":", "36", "PM");
    }

    @Test
    public void testNgram() {
        UniversalTokenizer tokenizer = new UniversalTokenizer();
        String text = "通以上 通以上";
        test(tokenizer, text, "通", "通以", "以", "以上", "上", "通", "通以", "以", "以上", "上");
        text = "非常感";
        test(tokenizer, text, "非", "非常", "常", "常感", "感");
        //text = "非\n\n常\n\n\n\n\n感";
	//test(tokenizer, text, "非", "非常", "常", "常感", "感");

        tokenizer = new UniversalTokenizer();
        tokenizer.setGenerateNgrams(false);
        test(tokenizer, text, "非", "常", "感");

        tokenizer.setGenerateUnigrams(false);
        test(tokenizer, text);

        tokenizer.setGenerateNgrams(true);
        test(tokenizer, text, "非常", "常感");
    }
}
