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
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Extracts the field value and translates it to an {@link OffsetDateTime} based on the specified {@link DateTimeFormatter}.
 * <p>
 * The formatter is supplied as a String to ensure it is tracked properly in the provenance.
 * <p>
 * Returns an empty optional if the date failed to parse.
 */
public class OffsetDateTimeExtractor extends SimpleFieldExtractor<OffsetDateTime> {
    private static final Logger logger = Logger.getLogger(OffsetDateTimeExtractor.class.getName());

    @Config(mandatory = true, description = "The expected date format.")
    private String dateTimeFormat;
    private DateTimeFormatter formatter;

    /**
     * for olcut
     */
    private OffsetDateTimeExtractor() {}

    /**
     * Constructs a date time extractor that emits an OffsetDateTime by applying the supplied format to the specified field.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write.
     * @param dateTimeFormat The date/time format (supplied to {@link DateTimeFormatter}.
     */
    public OffsetDateTimeExtractor(String fieldName, String metadataName, String dateTimeFormat) {
        super(fieldName, metadataName);
        this.dateTimeFormat = dateTimeFormat;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        if (dateTimeFormat != null) {
            formatter = DateTimeFormatter.ofPattern(dateTimeFormat);
        } else {
            throw new PropertyException("","dateTimeFormat", "Invalid Date/Time format string supplied");
        }
    }

    @Override
    public Class<OffsetDateTime> getValueType() {
        return OffsetDateTime.class;
    }

    @Override
    protected Optional<OffsetDateTime> extractField(String s) {
        try {
            return Optional.of(OffsetDateTime.parse(s, formatter));
        } catch (DateTimeParseException e) {
            logger.log(Level.WARNING, e.getParsedString());
            logger.log(Level.WARNING, String.format("Unable to parse date/time %s with formatter %s", s, dateTimeFormat));
            return Optional.empty();
        }
    }

    @Override
    public String toString() {
        return "OffsetDateTimeExtractor(fieldName=" + fieldName + ", metadataName=" + metadataName + ", dateTimeFormat=" + dateTimeFormat + ")";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "FieldExtractor");
    }
}
