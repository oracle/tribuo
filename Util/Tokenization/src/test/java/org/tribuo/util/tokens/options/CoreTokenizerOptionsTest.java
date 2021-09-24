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

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;
import org.tribuo.util.tokens.impl.NonTokenizer;
import org.tribuo.util.tokens.impl.ShapeTokenizer;
import org.tribuo.util.tokens.impl.SplitCharactersTokenizer;
import org.tribuo.util.tokens.impl.SplitPatternTokenizer;
import org.tribuo.util.tokens.universal.UniversalTokenizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CoreTokenizerOptionsTest extends TokenizerTestBase {

    @BeforeAll
    public static void setup() {
        Logger logger = Logger.getLogger(CoreTokenizerOptions.class.getName());
        logger.setLevel(Level.SEVERE);
    }

    @Test
    public void testBasic() {
        Tokenizer tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "BREAK_ITERATOR", "--bi-tokenizer-language-tag", "en"});
        assertTrue(tokenizer instanceof BreakIteratorTokenizer);
        assertEquals("en", ((BreakIteratorTokenizer) tokenizer).getLanguageTag());
        test(tokenizer, "Hello there!", "Hello", "there", "!");

        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "SPLIT_CHARACTERS", "--sc-tokenizer-split-characters", "(,)", "--sc-tokenizer-split-x-digits", ".,:"});
        assertTrue(tokenizer instanceof SplitCharactersTokenizer);
        test(tokenizer, "Hello there!", "Hello", "there!");
        test(tokenizer, "(Hello(to)(you) 1.23:45 78(89)", "Hello", "to", "you", "1.23:45", "78", "89");

        char escape = ConfigurationManager.CUR_ESCAPE_CHAR;
        String splitXChars = ".,:," + escape + ",";
        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "SPLIT_CHARACTERS", "--sc-tokenizer-split-characters", "(,)", "--sc-tokenizer-split-x-digits", splitXChars});
        assertTrue(tokenizer instanceof SplitCharactersTokenizer);
        test(tokenizer, "Hello there!", "Hello", "there!");
        test(tokenizer, "(Hello(to)(you) 1.23:45 78,89", "Hello", "to", "you", "1.23:45", "78,89");

        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "SPLIT_CHARACTERS"});
        assertTrue(tokenizer instanceof SplitCharactersTokenizer);
        test(tokenizer, "Hello there!", "Hello", "there");

        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "NON"});
        assertTrue(tokenizer instanceof NonTokenizer);
        test(tokenizer, "Hello there!", "Hello there!");

        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "SHAPE"});
        assertTrue(tokenizer instanceof ShapeTokenizer);
        test(tokenizer, "Hello there!", "Hello", "there", "!");
        test(tokenizer, "HelloThere123!", "Hello", "There", "123", "!");
        test(tokenizer, "ABCDefgHijkLmno!@#$123", "ABCDefg", "Hijk", "Lmno", "!", "@", "#", "$", "123");

        String regex = ConfigurationManager.IS_WINDOWS ? "\\s+" : "\\\\s+";
        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "SPLIT_PATTERN", "--sp-tokenizer-regex", regex});
        assertTrue(tokenizer instanceof SplitPatternTokenizer);
        test(tokenizer, "Hello there!", "Hello", "there!");

        tokenizer = createTokenizer(new String[]{"--core-tokenizer-type", "UNIVERSAL"});
        assertTrue(tokenizer instanceof UniversalTokenizer);
        test(tokenizer, "Hello there!", "Hello", "there");
        test(tokenizer, "HelloThere123!", "HelloThere123");
        test(tokenizer, "ABCDefgHijkLmno!@#$123", "ABCDefgHijkLmno", "123");

    }

    private Tokenizer createTokenizer(String[] args) {
        CoreTokenizerOptions options = new CoreTokenizerOptions();
        ConfigurationManager cm = new ConfigurationManager(args, options);
        Tokenizer tokenizer = options.getTokenizer();
        cm.close();
        return tokenizer;
    }

    @Test
    public void testComma() {
        char escape = ConfigurationManager.CUR_ESCAPE_CHAR;
        String[] args = new String[]{"--my-chars", "a," + escape + ",,b,c"};
        CommaOptions options = new CommaOptions();
        ConfigurationManager cm = new ConfigurationManager(args, options);
        cm.close();
        assertEquals(4, options.myChars.length);
        assertEquals('a', options.myChars[0]);
        assertEquals(',', options.myChars[1]);
        assertEquals('b', options.myChars[2]);
        assertEquals('c', options.myChars[3]);
    }

    public static class CommaOptions implements Options {

        /**
         * The characters.
         */
        @Option(longName = "my-chars", usage = "The characters.")
        public char[] myChars;

    }
}
