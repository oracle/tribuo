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

public class NonTokenizerTest extends TokenizerTestBase {

    @Test
    public void testBasic() {
        Tokenizer tokenizer = new NonTokenizer();
        test(tokenizer, "1.0n", "1.0n");
        test(tokenizer, "Hello there!", "Hello there!");
    }

    @Test
    public void testClone() {
        Tokenizer tokenizer = new NonTokenizer();
        testClones(tokenizer, "1.0n", "1.0n");
        testClones(tokenizer, "Hello there!", "Hello there!");
    }
}
