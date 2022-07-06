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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.text.DocumentPreprocessor;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * A simple document preprocessor which applies regular expressions to the input.
 */
public final class RegexPreprocessor implements DocumentPreprocessor {
    private List<Pattern> regexes;
    @Config(description = "A list of regular expressions in string format used to match the input", mandatory = true)
    private List<String> regexStrings;

    @Config(description = "A list of replacement strings which are used to replace the matches", mandatory = true)
    private List<String> replacements;

    /**
     * For OLCUT.
     */
    private RegexPreprocessor() {}

    /**
     * Construct a regex preprocessor.
     * @param regexStrings A list of strings containing regular expressions.
     * @param replacements A list of strings containing the replacements for matches
     *                     to the regular expressions in the input
     */
    public RegexPreprocessor(List<String> regexStrings, List<String> replacements) {
        this.replacements = replacements;
        this.regexStrings = regexStrings;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (regexStrings.size() != replacements.size()) {
            throw new PropertyException("", "regexStrings", String.format("The number of regex strings has to be the same as the number of replacement strings. " +
                    "%s regex string(s) and %s replacement string(s) were provided.", regexStrings.size(), replacements.size()));
        }
        regexes = new ArrayList<>(regexStrings.size());
        for (String regexString : regexStrings) {
            regexes.add(Pattern.compile(regexString));
        }
    }

    @Override
    public String processDoc(String doc) {

        for (int i=0 ; i<regexes.size(); i++) {
            doc = regexes.get(i).matcher(doc).replaceAll(replacements.get(i));
        }

        return doc;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"DocumentPreprocessor");
    }
}
