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

package org.tribuo.data.columnar.extractors;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Extracts the field value and translates it to a {@link LocalDate} based on the specified {@link DateTimeFormatter}.
 * <p>
 * Returns an empty optional if the date failed to parse.
 */
public class DateExtractor extends SimpleFieldExtractor<LocalDate> {
    private static final Logger logger = Logger.getLogger(DateExtractor.class.getName());

    @Config(mandatory = true, description = "The expected date format.")
    private String dateFormat;
    private DateTimeFormatter formatter;

    /**
     * for olcut
     */
    private DateExtractor() {}

    /**
     * Constructs a date extractor that emits a LocalDate by applying the supplied format to the specified field.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write.
     * @param dateFormat The date format (supplied to {@link DateTimeFormatter}.
     */
    public DateExtractor(String fieldName, String metadataName, String dateFormat) {
        super(fieldName, metadataName);
        this.dateFormat = dateFormat;
        postConfig();
    }

    /**
     * Constructs a date extractor that emits a LocalDate by applying the supplied format to the specified field.
     * <p>
     * Deprecated as it does not allow recovery of the formatter pattern for the provenance.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write.
     * @param formatter The date format (supplied to {@link DateTimeFormatter}.
     */
    @Deprecated
    public DateExtractor(String fieldName, String metadataName, DateTimeFormatter formatter) {
        super(fieldName, metadataName);
        this.formatter = formatter;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        if (dateFormat != null) {
            formatter = DateTimeFormatter.ofPattern(dateFormat);
        } else {
            formatter = DateTimeFormatter.BASIC_ISO_DATE;
        }
        if (metadataName == null || metadataName.isEmpty()) {
            metadataName = fieldName;
        }
    }

    @Override
    public Class<LocalDate> getValueType() {
        return LocalDate.class;
    }

    @Override
    protected Optional<LocalDate> extractField(String s) {
        try {
            return Optional.of(LocalDate.parse(s, formatter));
        } catch (DateTimeParseException e) {
            logger.log(Level.WARNING, e.getParsedString());
            logger.log(Level.WARNING, String.format("Unable to parse date %s with formatter %s", s, formatter.toString()));
            return Optional.empty();
        }
    }

    @Override
    public String toString() {
        return "DateExtractor(fieldName=" + fieldName + ", metadataName=" + metadataName + ", dateFormat=" + formatter.toString() + ")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "DateExtractor");
    }
}
