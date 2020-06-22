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

public class SplitCharactersTokenizerTest extends TokenizerTestBase {

    @Test
    public void testBasic() {
        Tokenizer tokenizer = new SplitCharactersTokenizer();

        test(tokenizer, "simple stupid test", "simple", "stupid", "test");

        // copied and modifed from DefaultTokenizerTest
        test(tokenizer, "1.0n", "1.0n");
        test(tokenizer, "1. 0n", "1", "0n");
        test(tokenizer, "a .10n", "a", "10n");
        test(tokenizer, "a ,10n", "a", "10n");
        test(tokenizer, "a, b, and c", "a", "b", "and", "c");

        // basic tests for each character
        test(tokenizer, "a-b", "a", "b");
        test(tokenizer, "a*b(c)d&e[f]g{h}i`j'k|l!m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m");
        test(tokenizer, "a.b 3.1 a.4 a1.2b c3.d4 5e.6f", "a", "b", "3.1", "a", "4", "a1.2b", "c3", "d4", "5e", "6f");
        test(tokenizer, "a/b 3/1 a/4 a1/2b c3/d4 5e/6f", "a", "b", "3/1", "a", "4", "a1/2b", "c3", "d4", "5e", "6f");
        test(tokenizer, "a,b 3,1 a,4 a1,2b c3,d4 5e,6f", "a", "b", "3,1", "a", "4", "a1,2b", "c3", "d4", "5e", "6f");

        // misc. tests
        test(tokenizer, "a&b(cd) a&&&b a(b][c]", "a", "b", "cd", "a", "b", "a", "b", "c");
        test(tokenizer, "a|bc|def* g* h (c'd`e)   {asdf}{fdsa} asdf{fdsa} ", "a", "bc", "def", "g", "h", "c", "d", "e",
                "asdf", "fdsa", "asdf", "fdsa");
        test(tokenizer, "$10,000.00 could be yours!", "$10,000.00", "could", "be", "yours");
        test(tokenizer, "1.2mm", "1.2mm");
        test(tokenizer, "1/2mm", "1/2mm");
        test(tokenizer, "3 oz.water", "3", "oz", "water");
        test(tokenizer, "3.", "3");
        test(tokenizer, "3 oz.", "3", "oz");

        test(tokenizer, "carmex oint.(gm) 15", "carmex", "oint", "gm", "15");
        test(tokenizer, "carmex oint(gm) 15", "carmex", "oint", "gm", "15");
        test(tokenizer, "carmex oint gm 15", "carmex", "oint", "gm", "15");

        tokenizer = new SplitCharactersTokenizer();
        test(tokenizer, "A*B(C)d&e[F]G{H}i`j'k|l!M A.B 3.1 a.4 a1.2B C3.d4 5E.6f", "A", "B", "C", "d", "e", "F", "G",
                "H", "i", "j", "k", "l", "M", "A", "B", "3.1", "a", "4", "a1.2B", "C3", "d4", "5E", "6f");
        tokenizer = new SplitCharactersTokenizer(null, null);
        test(tokenizer, "ab cd ef", "ab", "cd", "ef");

        tokenizer = new SplitCharactersTokenizer();
        test(tokenizer, "Washington, D.C.", "Washington", "D", "C");
        test(tokenizer, "U.S.", "U", "S");

        tokenizer = new SplitCharactersTokenizer(new char[]{'*', '(', ')', '&', '[', ']', '{', '}', '`',
                '\'', '|', ';', ':', '\\', '!', '-', '?'}, new char[]{',', '/'});
        test(tokenizer, "Washington, D.C.", "Washington", "D.C.");
        test(tokenizer, "U.S.", "U.S.");

    }

    @Test
    public void testClones() {
        Tokenizer tokenizer = new SplitCharactersTokenizer();

        testClones(tokenizer, "simple stupid test", "simple", "stupid", "test");

        // copied and modifed from DefaultTokenizerTest
        testClones(tokenizer, "1.0n", "1.0n");
        testClones(tokenizer, "1. 0n", "1", "0n");
        testClones(tokenizer, "a .10n", "a", "10n");
        testClones(tokenizer, "a ,10n", "a", "10n");
        testClones(tokenizer, "a, b, and c", "a", "b", "and", "c");

        // basic testCloness for each character
        testClones(tokenizer, "a-b", "a", "b");
        testClones(tokenizer, "a*b(c)d&e[f]g{h}i`j'k|l!m", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m");
        testClones(tokenizer, "a.b 3.1 a.4 a1.2b c3.d4 5e.6f", "a", "b", "3.1", "a", "4", "a1.2b", "c3", "d4", "5e", "6f");
        testClones(tokenizer, "a/b 3/1 a/4 a1/2b c3/d4 5e/6f", "a", "b", "3/1", "a", "4", "a1/2b", "c3", "d4", "5e", "6f");
        testClones(tokenizer, "a,b 3,1 a,4 a1,2b c3,d4 5e,6f", "a", "b", "3,1", "a", "4", "a1,2b", "c3", "d4", "5e", "6f");

        // misc. testCloness
        testClones(tokenizer, "a&b(cd) a&&&b a(b][c]", "a", "b", "cd", "a", "b", "a", "b", "c");
        testClones(tokenizer, "a|bc|def* g* h (c'd`e)   {asdf}{fdsa} asdf{fdsa} ", "a", "bc", "def", "g", "h", "c", "d", "e",
                "asdf", "fdsa", "asdf", "fdsa");
        testClones(tokenizer, "$10,000.00 could be yours!", "$10,000.00", "could", "be", "yours");
        testClones(tokenizer, "1.2mm", "1.2mm");
        testClones(tokenizer, "1/2mm", "1/2mm");
        testClones(tokenizer, "3 oz.water", "3", "oz", "water");
        testClones(tokenizer, "3.", "3");
        testClones(tokenizer, "3 oz.", "3", "oz");

        testClones(tokenizer, "carmex oint.(gm) 15", "carmex", "oint", "gm", "15");
        testClones(tokenizer, "carmex oint(gm) 15", "carmex", "oint", "gm", "15");
        testClones(tokenizer, "carmex oint gm 15", "carmex", "oint", "gm", "15");

        tokenizer = new SplitCharactersTokenizer();
        testClones(tokenizer, "A*B(C)d&e[F]G{H}i`j'k|l!M A.B 3.1 a.4 a1.2B C3.d4 5E.6f", "A", "B", "C", "d", "e", "F", "G",
                "H", "i", "j", "k", "l", "M", "A", "B", "3.1", "a", "4", "a1.2B", "C3", "d4", "5E", "6f");
        tokenizer = new SplitCharactersTokenizer(null, null);
        testClones(tokenizer, "ab cd ef", "ab", "cd", "ef");
    }
}
