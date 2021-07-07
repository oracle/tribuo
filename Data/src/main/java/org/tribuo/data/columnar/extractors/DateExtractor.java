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

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.Locale;
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

    @Config(mandatory = false, description = "Sets the locale language.")
    private String localeLanguage = null;

    @Config(mandatory = false, description = "Sets the locale country.")
    private String localeCountry = null;

    /**
     * for olcut
     */
    private DateExtractor() {}

    /**
     * Constructs a date extractor that emits a LocalDate by applying the supplied format to the specified field.
     * <p>
     * Uses the system locale for backwards compatibility with 4.0 and 4.1.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write.
     * @param dateFormat The date format (supplied to {@link DateTimeFormatter}.
     */
    public DateExtractor(String fieldName, String metadataName, String dateFormat) {
        this(fieldName,metadataName,dateFormat,null,null);
    }

    /**
     * Constructs a date extractor that emits a LocalDate by applying the supplied format to the specified field.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write to.
     * @param dateFormat The date format (supplied to {@link DateTimeFormatter}.
     * @param localeLanguage The locale language.
     * @param localeCountry The locale country.
     */
    public DateExtractor(String fieldName, String metadataName, String dateFormat, String localeLanguage, String localeCountry) {
        super(fieldName, metadataName);
        this.dateFormat = dateFormat;
        this.localeCountry = localeCountry;
        this.localeLanguage = localeLanguage;
        postConfig();
    }

    /**
     * Constructs a date extractor that emits a LocalDate by applying the supplied format to the specified field.
     * <p>
     * Deprecated as it does not allow recovery of the formatter pattern for the provenance.
     * <p>
     * Uses the system locale for backwards compatibility with 4.0 and 4.1.
     * @param fieldName The field to read.
     * @param metadataName The metadata field to write.
     * @param formatter The date format (supplied to {@link DateTimeFormatter}.
     */
    @Deprecated
    public DateExtractor(String fieldName, String metadataName, DateTimeFormatter formatter) {
        super(fieldName, metadataName);
        this.formatter = formatter;
        this.dateFormat = formatter.toString();
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
        if (dateFormat != null) {
            try {
                formatter = DateTimeFormatter.ofPattern(dateFormat,locale);
            } catch (IllegalArgumentException e) {
                throw new PropertyException(e,"","dateFormat","dateFormat could not be parsed by DateTimeFormatter");
            }
        } else {
            formatter = DateTimeFormatter.BASIC_ISO_DATE;
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
        return "DateExtractor(" +
                "fieldName='" + fieldName + '\'' +
                ", metadataName='" + metadataName + '\'' +
                ", dateFormat=" + formatter +
                ", localeLanguage='" + localeLanguage + '\'' +
                ", localeCountry='" + localeCountry + '\'' +
                ')';
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "DateExtractor");
    }
}
