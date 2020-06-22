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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.universal.UniversalTokenizer;

import java.io.File;
import java.io.IOException;
import java.util.Locale;

import static org.tribuo.util.tokens.TokenizerTestWrapper.serializeAndDeserialize;


public class TokenizerSerializationTest extends TokenizerTestBase {

    private File f;

    @BeforeEach
    public void setUp() throws IOException {
        f = File.createTempFile("serialized-tokenizer", ".ser", new File("target"));
        f.deleteOnExit();
    }

    @Test
    public void testSerializeDeserialize() throws IOException, ClassNotFoundException {
        //SplitPatternTokenizer
        Tokenizer tokenizer = new SplitPatternTokenizer();
        test(tokenizer, "a, b, and c", "a", "b", "and", "c");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "a, b, and c", "a", "b", "and", "c");
        tokenizer = new SplitPatternTokenizer("\\s+");
        test(tokenizer, "a   b c", "a", "b", "c");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "a   b c", "a", "b", "c");
        //SplitCharactersTokenizer
        tokenizer = new SplitCharactersTokenizer();
        test(tokenizer, "a*b(c)d&e[f]g{h}i`j'k|l!m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "a*b(c)d&e[f]g{h}i`j'k|l!m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m");
        tokenizer = new SplitCharactersTokenizer(new char[]{'*', '(', ')', '&', '[', ']', '{', '}', '`', '\'', '|', ';', ':', '\\', '!', '-', '?'}, new char[]{',', '/'});
        test(tokenizer, "Washington, D.C.", "Washington", "D.C.");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "Washington, D.C.", "Washington", "D.C.");
        //ShapeTokenizer
        tokenizer = new ShapeTokenizer();
        test(tokenizer, "ABCIndustries AB123 A1B2", "ABCIndustries", "AB", "123", "A", "1", "B", "2");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "ABCIndustries AB123 A1B2", "ABCIndustries", "AB", "123", "A", "1", "B", "2");
        //NonTokenizer
        tokenizer = new NonTokenizer();
        test(tokenizer, "Hello there!", "Hello there!");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "Hello there!", "Hello there!");
        //BreakIteratorTokenizer
        tokenizer = new BreakIteratorTokenizer(Locale.US);
        test(tokenizer, "http://www.acme.com/SH55126545/VD55177927", "http", ":", "/", "/", "www.acme.com", "/", "SH55126545", "/", "VD55177927");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "http://www.acme.com/SH55126545/VD55177927", "http", ":", "/", "/", "www.acme.com", "/", "SH55126545", "/", "VD55177927");
        //UniversalTokenizer
        tokenizer = new UniversalTokenizer(true);
        test(tokenizer, "4:36 PM", "4", ":", "36", "PM");
        tokenizer = serializeAndDeserialize(f, tokenizer);
        test(tokenizer, "4:36 PM", "4", ":", "36", "PM");
    }

}
