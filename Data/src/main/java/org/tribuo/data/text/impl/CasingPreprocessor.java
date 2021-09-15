/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.text.DocumentPreprocessor;

/**
 * A document preprocessor which uppercases or lowercases the input.
 */
public class CasingPreprocessor implements DocumentPreprocessor {

    /**
     * The possible casing operations.
     */
    public enum CasingOperation {
        /**
         * Lowercase the input text.
         */
        LOWERCASE,
        /**
         * Uppercase the input text.
         */
        UPPERCASE;

        /**
         * Apply the appropriate casing operation.
         * @param input The input to transform.
         * @return The transformed input.
         */
        public String applyCase(String input) {
            switch (this) {
                case UPPERCASE:
                    return input.toUpperCase();
                case LOWERCASE:
                    return input.toLowerCase();
                default:
                    throw new IllegalStateException("Unexpected enum value " + this.toString());
            }
        }
    }

    @Config(description="Which casing operation to apply.")
    private CasingOperation op = CasingOperation.LOWERCASE;

    /**
     * For OLCUT.
     */
    private CasingPreprocessor() {}

    /**
     * Construct a casing preprocessor.
     * @param op The operation to apply.
     */
    public CasingPreprocessor(CasingOperation op) {
        this.op = op;
    }

    @Override
    public String processDoc(String doc) {
        return op.applyCase(doc);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"DocumentPreprocessor");
    }

}
