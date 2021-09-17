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

package org.tribuo.data.columnar.processors.field;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FieldProcessor;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * A {@link FieldProcessor} which applies a regex to a field and generates {@link ColumnarFeature}s based on the matches.
 */
public class RegexFieldProcessor implements FieldProcessor {
    private static final Logger logger = Logger.getLogger(RegexFieldProcessor.class.getName());

    private Pattern regex;

    @Config(mandatory = true,description="Regex to apply to the field.")
    private String regexString;

    @Config(mandatory = true,description="The field name to read.")
    private String fieldName;

    @Config(mandatory = true,description="Matching mode.")
    private EnumSet<Mode> modes;

    /**
     * Matching mode.
     */
    public enum Mode {
        /**
         * Triggers feature generation if the whole string matches.
         */
        MATCH_ALL,
        /**
         * Triggers feature generation if the string contains a match.
         */
        MATCH_CONTAINS,
        /**
         * Triggers feature generation for each matching group in the string.
         */
        GROUPS
    }

    /**
     * For olcut.
     */
    private RegexFieldProcessor() {}

    /**
     * Constructs a field processor which emits features when the field value matches the supplied regex.
     * @param fieldName The field name to read.
     * @param regex The regex to use for matching.
     * @param modes The matching mode.
     */
    public RegexFieldProcessor(String fieldName, Pattern regex, EnumSet<Mode> modes) {
        this.regex = regex;
        this.fieldName = fieldName;
        this.regexString = regex.pattern();
        this.modes = modes;
    }

    /**
     * Constructs a field processor which emits features when the field value matches the supplied regex.
     * <p>
     * The regex is compiled on construction.
     * @param fieldName The field name to read.
     * @param regex The regex to use for matching.
     * @param modes The matching mode.
     */
    public RegexFieldProcessor(String fieldName, String regex, EnumSet<Mode> modes) {
        this(fieldName,Pattern.compile(regex),modes);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        this.regex = Pattern.compile(regexString);
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public List<ColumnarFeature> process(String value) {
        List<ColumnarFeature> features = new ArrayList<>();
        Matcher m = regex.matcher(value);
        for (Mode mode : modes) {
            switch (mode) {
                case MATCH_ALL:
                    if (m.matches()) {
                        features.add(new ColumnarFeature(fieldName,"MATCHES_ALL", 1.0));
                    }
                    break;
                case MATCH_CONTAINS:
                    if (m.find()) {
                        features.add(new ColumnarFeature(fieldName,"CONTAINS_MATCH", 1.0));
                    }
                    break;
                case GROUPS:
                    int i = 0;
                    while (m.find()) {
                        i++;
                        features.add(new ColumnarFeature(fieldName, "GROUPS(" + m.group(i) + ")", 1.0));
                    }
                    break;
            }
        }
        return features;
    }

    @Override
    public GeneratedFeatureType getFeatureType() {
        return GeneratedFeatureType.CATEGORICAL;
    }

    @Override
    public RegexFieldProcessor copy(String newFieldName) {
        return new RegexFieldProcessor(newFieldName, regex, EnumSet.copyOf(modes));
    }

    @Override
    public String toString() {
        return "RegexFieldProcessor(fieldName="+getFieldName()+",modes=" + modes.stream().map(Mode::name).sorted().collect(Collectors.joining(":"))+')';
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldProcessor");
    }
}
