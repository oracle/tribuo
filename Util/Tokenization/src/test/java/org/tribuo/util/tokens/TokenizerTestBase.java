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

package org.tribuo.util.tokens;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Base class for tokenizer testing, checks that the expected tokens are produced when a snippet is tokenized.
 */
public class TokenizerTestBase {

    protected void test(Tokenizer tokenizer, String text, String... expectedTokens) {
        List<String> actualSplits = tokenizer.split(text);
        List<Token> actualTokens = tokenizer.tokenize(text);
        String message = "\"" + text + "\" actual split: " + actualSplits + " actual tokens: " + actualTokens;

        assertEquals(expectedTokens.length, actualSplits.size(), message + ", length of splits not equal -");
        assertEquals(expectedTokens.length, actualTokens.size(), message + ", length of tokens not equal -");

        for (int i = 0; i < expectedTokens.length; i++) {
            assertEquals(expectedTokens[i], actualSplits.get(i), message + ", split strings don't match -");
            assertEquals(expectedTokens[i], actualTokens.get(i).text, message + ", token strings don't match -");
            assertEquals(expectedTokens[i], text.substring(actualTokens.get(i).start, actualTokens.get(i).end), message + ", start & end values don't produce correct substring (start="
                    + actualTokens.get(i).start + ",end=" + actualTokens.get(i).end + ")-");
        }
    }

    protected void testClones(Tokenizer tokenizer, String text, String... expectedTokens) {
        Tokenizer otherTokenizer;
        try {
            otherTokenizer = tokenizer.clone();
            try {
                otherTokenizer.advance();
                fail();
            } catch (IllegalStateException e) {
            }
            try {
                otherTokenizer.getText();
                fail();
            } catch (IllegalStateException e) {
            }

            List<String> actualSplits = new ArrayList<>();
            List<Token> actualTokens = new ArrayList<>();

            List<String> otherActualSplits = new ArrayList<>();
            List<Token> otherActualTokens = new ArrayList<>();

            tokenizer.reset(text);
            otherTokenizer.reset(text);
            while (tokenizer.advance()) {
                otherTokenizer.advance();
                actualSplits.add(tokenizer.getText());
                otherActualSplits.add(otherTokenizer.getText());
            }

            tokenizer.reset(text);
            otherTokenizer.reset(text);
            while (tokenizer.advance()) {
                actualTokens.add(tokenizer.getToken());
                otherTokenizer.advance();
                otherActualTokens.add(otherTokenizer.getToken());
            }

            String message = "\"" + text + "\" actual split: " + actualSplits + " actual tokens: " + actualTokens;

            assertEquals(expectedTokens.length, actualSplits.size(), message + ", length of splits not equal -");
            assertEquals(expectedTokens.length, actualTokens.size(), message + ", length of tokens not equal -");

            assertEquals(expectedTokens.length, otherActualSplits.size(), message + ", length of splits not equal in clone -");
            assertEquals(expectedTokens.length, otherActualTokens.size(), message + ", length of tokens not equal in clone -");

            for (int i = 0; i < expectedTokens.length; i++) {
                assertEquals(expectedTokens[i], actualSplits.get(i), message + ", split strings don't match -");
                assertEquals(expectedTokens[i], actualTokens.get(i).text, message + ", token strings don't match -");
                assertEquals(expectedTokens[i], text.substring(actualTokens.get(i).start, actualTokens.get(i).end), message + ", start & end values don't produce correct substring (start="
                        + actualTokens.get(i).start + ",end=" + actualTokens.get(i).end + ")-");

                assertEquals(expectedTokens[i], otherActualSplits.get(i), message + ", split strings don't match in clone -");
                assertEquals(expectedTokens[i], otherActualTokens.get(i).text, message + ", token strings don't match in clone -");
                assertEquals(expectedTokens[i], text.substring(otherActualTokens.get(i).start, otherActualTokens.get(i).end), message + ", start & end values don't produce correct substring in clone (start="
                        + otherActualTokens.get(i).start + ",end=" + otherActualTokens.get(i).end + ")-");
            }
        } catch (CloneNotSupportedException e) {
            fail("Failed to clone tokenizer " + e.getMessage());
        }
    }

}
