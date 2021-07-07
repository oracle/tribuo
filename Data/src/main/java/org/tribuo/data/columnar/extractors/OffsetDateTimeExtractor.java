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
import java.util.Locale;
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

    @Config(mandatory = false, description = "The locale language.")
    private String localeLanguage = null;

    @Config(mandatory = false, description = "The locale country.")
    private String localeCountry = null;

    /**
     * for olcut
     */
    private OffsetDateTimeExtractor() {}

    /**
     * Constructs a date time extractor that emits an OffsetDateTime by applying the supplied format to the specified field.
     * <p>
     * Uses the system locale for backwards compatibility with 4.0 and 4.1.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write.
     * @param dateTimeFormat The date/time format (supplied to {@link DateTimeFormatter}.
     */
    public OffsetDateTimeExtractor(String fieldName, String metadataName, String dateTimeFormat) {
        this(fieldName,metadataName,dateTimeFormat,null,null);
    }

    /**
     * Constructs a date time extractor that emits an OffsetDateTime by applying the supplied format to the specified field.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write to.
     * @param dateTimeFormat The date format (supplied to {@link DateTimeFormatter}.
     * @param localeLanguage The locale language.
     * @param localeCountry The locale country.
     */
    public OffsetDateTimeExtractor(String fieldName, String metadataName, String dateTimeFormat, String localeLanguage, String localeCountry) {
        super(fieldName, metadataName);
        this.dateTimeFormat = dateTimeFormat;
        this.localeCountry = localeCountry;
        this.localeLanguage = localeLanguage;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        super.postConfig();
        Locale locale;
        if ((localeLanguage == null) && (localeCountry == null)) {
            locale = Locale.getDefault(Locale.Category.FORMAT);
        } else if (localeLanguage == null) {
            throw new PropertyException("","localeLanguage","Must supply both localeLanguage and localeCountry when setting the locale.");
        } else if (localeCountry == null) {
            throw new PropertyException("","localeCountry","Must supply both localeLanguage and localeCountry when setting the locale.");
        } else {
            locale = new Locale(localeLanguage,localeCountry);
        }
        if (dateTimeFormat != null) {
            try {
                formatter = DateTimeFormatter.ofPattern(dateTimeFormat,locale);
            } catch (IllegalArgumentException e) {
                throw new PropertyException(e,"","dateTimeFormat","dateTimeFormat could not be parsed by DateTimeFormatter");
            }
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
        return "OffsetDateTimeExtractor(" +
                "fieldName='" + fieldName + '\'' +
                ", metadataName='" + metadataName + '\'' +
                ", dateTimeFormat='" + dateTimeFormat + '\'' +
                ", localeLanguage='" + localeLanguage + '\'' +
                ", localeCountry='" + localeCountry + '\'' +
                ')';
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "FieldExtractor");
    }
}
