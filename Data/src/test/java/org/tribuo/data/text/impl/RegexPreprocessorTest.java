/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.data.text.impl;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.Test;
import java.util.ArrayList;


import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;


public class RegexPreprocessorTest {
    @Test
    public void testUnequalLengths() {
        ArrayList<String> replacements = new ArrayList<String>();
        replacements.add(" ");
        ArrayList<String> regexStrings = new ArrayList<String>();
        regexStrings.add("[^A-Za-z]");
        regexStrings.add("[\\p{Punct}]");
        assertThrows(PropertyException.class, () -> new RegexPreprocessor(regexStrings, replacements));
    }

    @Test
    public void testOrder() {
        String doc = "sample document";
        ArrayList<String> replacements = new ArrayList<String>();
        replacements.add("*");
        replacements.add("");
        ArrayList<String> regexStrings = new ArrayList<String>();
        regexStrings.add("[!A-Za-z]");
        regexStrings.add("[\\p{Punct}]");
        RegexPreprocessor regexPreprocessor = new RegexPreprocessor(regexStrings, replacements);
        assertEquals(" ", regexPreprocessor.processDoc(doc));

    }
}
