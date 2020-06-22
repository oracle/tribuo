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

public class ShapeTokenizerTest extends TokenizerTestBase {

    @Test
    public void testTokenizer() {
        ShapeTokenizer tokenizer = new ShapeTokenizer();
        test(tokenizer, "");
        test(tokenizer, "This is a test.", "This", "is", "a", "test", ".");
        test(tokenizer, "CamelCase", "Camel", "Case");
        test(tokenizer, "ABCIndustries AB123 A1B2", "ABCIndustries", "AB", "123", "A", "1", "B", "2");
        test(tokenizer, "PUNJANA TEABAGS 160S PMP ú2.99", "PUNJANA", "TEABAGS", "160", "S", "PMP", "ú", "2", ".", "99");
        test(tokenizer, "$48 million and $41 million for Q1 2010 and Q1 2009.", "$", "48", "million", "and", "$", "41", "million", "for", "Q", "1", "2010", "and", "Q", "1", "2009", ".");
        test(tokenizer, "100152968 'M'RED LABEL TEA BAGS 240'S", "100152968", "'", "M", "'", "RED", "LABEL", "TEA", "BAGS", "240", "'", "S");
        test(tokenizer, " test TestCase5", "test", "Test", "Case", "5");
        test(tokenizer, "1234ABCD#@!%", "1234", "ABCD", "#", "@", "!", "%");
    }

    @Test
    public void testClone() {
        ShapeTokenizer tokenizer = new ShapeTokenizer();
        testClones(tokenizer, "");
        testClones(tokenizer, "This is a test.", "This", "is", "a", "test", ".");
        testClones(tokenizer, "CamelCase", "Camel", "Case");
        testClones(tokenizer, "ABCIndustries AB123 A1B2", "ABCIndustries", "AB", "123", "A", "1", "B", "2");
        testClones(tokenizer, "PUNJANA TEABAGS 160S PMP ú2.99", "PUNJANA", "TEABAGS", "160", "S", "PMP", "ú", "2", ".", "99");
        testClones(tokenizer, "$48 million and $41 million for Q1 2010 and Q1 2009.", "$", "48", "million", "and", "$", "41", "million", "for", "Q", "1", "2010", "and", "Q", "1", "2009", ".");
        testClones(tokenizer, "100152968 'M'RED LABEL TEA BAGS 240'S", "100152968", "'", "M", "'", "RED", "LABEL", "TEA", "BAGS", "240", "'", "S");
        testClones(tokenizer, " test TestCase5", "test", "Test", "Case", "5");
        testClones(tokenizer, "1234ABCD#@!%", "1234", "ABCD", "#", "@", "!", "%");
    }
}
