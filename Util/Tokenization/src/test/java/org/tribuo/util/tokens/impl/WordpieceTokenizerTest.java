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

package org.tribuo.util.tokens.impl;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.params.provider.Arguments.arguments;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.tribuo.util.tokens.impl.wordpiece.Wordpiece;
import org.tribuo.util.tokens.impl.wordpiece.WordpieceBasicTokenizer;
import org.tribuo.util.tokens.impl.wordpiece.WordpieceTokenizer;

import com.oracle.labs.mlrg.olcut.util.IOUtil;

/**
 * To test this tokenizer we threw a bunch of text segments drawn from Wikipedia from lots of different languages
 * and generated regression data using the referenced python implementation.  The regression data can be found in
 * <code>src/test/resources</code> as <code>/org/tribuo/util/tokens/impl/test/regression-text_bert-base-uncased.txt</code> and
 * <code>/org/tribuo/util/tokens/impl/test/regression-text_bert-base-uncased_failes.txt</code>.  The former is used in the regression
 * test defined below and each line represents a passing regression test.  The latter represents the outstanding and known 
 * regression test failures.  Most of these involve texts that include Arabic and other
 * non-latin scripts that generate so many "[UNK]" tokens for an English-based
 * BPE vocabulary as to render the discrepancies as practically meaningless.  
 */
public class WordpieceTokenizerTest extends TokenizerTestBase {

    public static Stream<Arguments> testWordpiece() throws Exception {
        Wordpiece wordpiece = new Wordpiece("src/test/resources/co/huggingface/bert-base-uncased.txt");
        Tokenizer tokenizer = new WordpieceTokenizer(wordpiece, new WordpieceBasicTokenizer(), true, true, Collections.emptySet());

        return Stream.of(arguments(tokenizer, "", Collections.emptyList()),
                arguments(tokenizer, "test", Arrays.asList("test")),
                arguments(tokenizer, "one-year-old", Arrays.asList("one", "-", "year", "-", "old")),
                arguments(tokenizer, "partyg", Arrays.asList("party", "##g")),
                arguments(tokenizer, "whitecastleburgers", Arrays.asList("white", "##castle", "##burg", "##ers")),
                arguments(tokenizer, "hello world!", Arrays.asList("hello", "world", "!")),
                arguments(tokenizer, "prince humperdink", Arrays.asList("prince", "hum", "##per", "##din", "##k")),
                arguments(tokenizer, "prince४०८", Arrays.asList("[UNK]")),
                arguments(tokenizer, "the ४०८", Arrays.asList("the", "[UNK]")),
                arguments(tokenizer, "      ", Collections.emptyList()),
                arguments(tokenizer, "官", Arrays.asList("[UNK]")),
                arguments(tokenizer, "𧩙", Arrays.asList("[UNK]")),
                arguments(tokenizer, "官𧩙", Arrays.asList("[UNK]", "[UNK]")));
        
    }

    @ParameterizedTest
    @MethodSource
    public void testWordpiece(Tokenizer tokenizer, String text, List<String> expectedTokens) {
        List<String> actualTokens = tokenizer.split(text);
        Assertions.assertEquals(expectedTokens, actualTokens);
    }

    
    public static Stream<Arguments> testNeverSplit() throws Exception {
        Wordpiece wordpiece = new Wordpiece("src/test/resources/co/huggingface/bert-base-uncased.txt");
        Set<String> neverSplits = new HashSet<>(Arrays.asList("one-year-old", "$4.50"));
        
        Tokenizer tokenizer = new WordpieceTokenizer(wordpiece, new WordpieceBasicTokenizer(), true, true, neverSplits);

        return Stream.of(
                arguments(tokenizer, "", Collections.emptyList())
                , arguments(tokenizer, "one-year-old", Arrays.asList("one-year-old"))
                , arguments(tokenizer, "$4.50", Arrays.asList("$4.50"))
                , arguments(tokenizer, "$5.50", Arrays.asList("$", "5", ".", "50"))
                );
        
    }

    @ParameterizedTest
    @MethodSource
    public void testNeverSplit(Tokenizer tokenizer, String text, List<String> expectedTokens) {
        List<String> actualTokens = tokenizer.split(text);
        Assertions.assertEquals(expectedTokens, actualTokens);
    }

    @Test
    public void testClone() throws CloneNotSupportedException {
        Wordpiece wordpiece = new Wordpiece("src/test/resources/co/huggingface/bert-base-uncased.txt");
        Set<String> neverSplits = new HashSet<>(Arrays.asList("one-year-old", "$4.50"));
        Tokenizer tokenizer = new WordpieceTokenizer(wordpiece, new WordpieceBasicTokenizer(), true, true, neverSplits);
        tokenizer = tokenizer.clone();
        
        List<String> actualTokens = tokenizer.split("1.0N");
        Assertions.assertEquals(Arrays.asList("1",".","0", "##n"), actualTokens);
        actualTokens = tokenizer.split("1.0ñ");
        Assertions.assertEquals(Arrays.asList("1",".","0", "##n"), actualTokens);

        tokenizer = new WordpieceTokenizer(wordpiece, new WordpieceBasicTokenizer(), false, false, neverSplits);
        tokenizer = tokenizer.clone();
        
        actualTokens = tokenizer.split("1.0N");
        Assertions.assertEquals(Arrays.asList("1",".", "[UNK]"), actualTokens);

    }
    
    @Test
    public void regressionTest() throws Exception {
        Wordpiece wordpiece = new Wordpiece("src/test/resources/co/huggingface/bert-base-uncased.txt");
        Tokenizer tokenizer = new WordpieceTokenizer(wordpiece, new WordpieceBasicTokenizer(), true, true, Collections.emptySet());

        List<String> lines = IOUtil
                .getLines("src/test/resources/org/tribuo/util/tokens/impl/test/regression-text_bert-base-uncased.txt");

        final AtomicInteger progress = new AtomicInteger(0);
        assertAll(lines.stream().map(line -> {
            int p = progress.incrementAndGet();
            String[] data = line.split("\\t");
            String text = data[0];
            String[] expectedTokens = new String[data.length - 1];
            System.arraycopy(data, 1, expectedTokens, 0, data.length - 1);
            List<String> expectedTokensList = Arrays.asList(expectedTokens);
            List<String> actualTokensList = tokenizer.split(text);
            return () -> assertEquals(expectedTokensList, actualTokensList, "line="+p+": "+text);
        }));
    }
}