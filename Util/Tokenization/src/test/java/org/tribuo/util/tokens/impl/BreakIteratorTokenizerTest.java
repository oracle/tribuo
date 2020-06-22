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

import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.junit.jupiter.api.Test;

import java.util.Locale;

public class BreakIteratorTokenizerTest extends TokenizerTestBase {

    @Test
    public void testBasic() {
        Tokenizer tokenizer = new BreakIteratorTokenizer(Locale.US);
        test(tokenizer, "1.0n", "1.0n");
        test(tokenizer, "Hello there!", "Hello", "there", "!");
    }

    @Test
    public void testClone() {
        BreakIteratorTokenizer tokenizer = new BreakIteratorTokenizer(Locale.US);
        testClones(tokenizer, "1.0n", "1.0n");
        testClones(tokenizer, "Hello there!", "Hello", "there", "!");
    }

    /**
     * This test demonstrates a surprising fact about the BreakIterator's tokenization
     */
    @Test
    public void testQuotes() {
        Tokenizer tokenizer = new BreakIteratorTokenizer(Locale.US);
        test(tokenizer, "\"a!a!a\"", "\"", "a", "!", "a", "!", "a", "\"");
        test(tokenizer, "\"a\"a\"a\"", "\"", "a\"a\"a", "\"");  //{"a"a"a"} is tokenized to {" a"a"a "}
    }

    @Test
    public void testURLBug() {
        Tokenizer tokenizer = new BreakIteratorTokenizer(Locale.US);
        test(tokenizer, "http://www.acme.com/SH55126545/VD55177927", "http", ":", "/", "/", "www.acme.com", "/", "SH55126545", "/", "VD55177927");
        test(tokenizer, "Development", "Development");
        test(tokenizer, "Mozilla/5.0", "Mozilla", "/", "5.0");
        test(tokenizer, "5.0", "5.0");
    }


    @Test
    public void testPunctuationBug() {
        Tokenizer tokenizer = new BreakIteratorTokenizer(Locale.US);
        test(tokenizer, "&&$#", "&", "&", "$", "#");
        test(tokenizer, "testQAtesting testing", "testQAtesting", "testing");
    }
}
