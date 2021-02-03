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

import org.junit.jupiter.api.Test;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.TokenizerTestBase;
import org.tribuo.util.tokens.impl.wordpiece.WordpieceBasicTokenizer;

public class WordpieceBasicTokenizerTest extends TokenizerTestBase {

    @Test
    public void testBasic() {
        Tokenizer tokenizer = new WordpieceBasicTokenizer();
        test(tokenizer, "a b  c\nd\t\te \t\r\nf", "a", "b", "c", "d", "e", "f");
        test(tokenizer, "");
        test(tokenizer, " \t\r\n");
        test(tokenizer, " \t\r\na", "a");
        test(tokenizer, "a \t\r\n", "a");
        test(tokenizer, "abcd", "abcd");
        test(tokenizer, "abcd efgh", "abcd", "efgh");
        test(tokenizer, "hello world!", "hello", "world", "!");
        test(tokenizer, "hello-world!!!goodbye  earth. p", "hello", "-", "world", "!", "!", "!", "goodbye", "earth", ".", "p");
        test(tokenizer, "8 km", "8", "km");
        test(tokenizer, "238.8 km", "238", ".", "8", "km");
        test(tokenizer, "金泰均", "金", "泰", "均");
    }

}
