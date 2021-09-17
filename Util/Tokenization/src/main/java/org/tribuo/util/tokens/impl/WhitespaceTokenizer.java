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

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 * A simple tokenizer that splits on whitespace.  This tokenizer does not create
 * tokens that correspond to whitespace - only those spans of text delimited by
 * whitespace.  For example, the text "a b" will result in two tokens "a" and "b". 
 */
public class WhitespaceTokenizer extends SplitFunctionTokenizer {

    /**
     * The splitting function for whitespace, using {@link Character#isWhitespace(char)}.
     */
    public static final SplitFunction whitespaceSplitCharacterFunction = (codepoint, index,
            cs) -> Character.isWhitespace(codepoint) ? SplitResult.SPLIT_AT : SplitResult.NO_SPLIT_WORD;

    /**
     * Constructs a tokenizer that splits on whitespace.
     */
    public WhitespaceTokenizer() {
        super(whitespaceSplitCharacterFunction);
    }
    
    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public WhitespaceTokenizer clone() {
        return new WhitespaceTokenizer();
    }

}
