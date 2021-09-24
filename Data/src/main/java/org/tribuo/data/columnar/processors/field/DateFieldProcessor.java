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

package org.tribuo.data.columnar.processors.field;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.columnar.ColumnarFeature;
import org.tribuo.data.columnar.FieldProcessor;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.time.temporal.IsoFields;
import java.time.temporal.WeekFields;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.function.ToIntFunction;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Processes a column that contains a date value. The features are named
 * based on the column name and the feature type extracted.
 * <p>
 * This class uses {@link LocalDate}
 * <p>
 * Returns an empty list if the date failed to parse or was empty.
 */
public final class DateFieldProcessor implements FieldProcessor {

    private static final Logger logger = Logger.getLogger(DateFieldProcessor.class.getName());

    /**
     * The types of date features which can be extracted.
     */
    public enum DateFeatureType {
        /**
         * The day of the week in ISO 8601.
         */
        DAY_OF_WEEK((LocalDate l) -> l.getDayOfWeek().getValue()),
        /**
         * The week of the year in ISO 8601.
         */
        WEEK_OF_YEAR((LocalDate l) -> l.get(WeekFields.ISO.weekOfWeekBasedYear())),
        /**
         * The week of the month, as defined by ISO 8601 semantics for week of the year.
         */
        WEEK_OF_MONTH((LocalDate l) -> l.get(WeekFields.ISO.weekOfMonth())),
        /**
         * The day of the year.
         */
        DAY_OF_YEAR(LocalDate::getDayOfYear),
        /**
         * The parity of the day of the year.
         */
        EVEN_OR_ODD_DAY((LocalDate l) -> l.getDayOfYear() % 2),
        /**
         * The parity of the week of the year as defined by ISO 8601.
         */
        EVEN_OR_ODD_WEEK((LocalDate l) -> l.get(WeekFields.ISO.weekOfWeekBasedYear()) % 2),
        /**
         * The parity of the month.
         */
        EVEN_OR_ODD_MONTH((LocalDate l) -> l.getMonthValue() % 2),
        /**
         * The parity of the year.
         */
        EVEN_OR_ODD_YEAR((LocalDate l) -> l.getYear() % 2),
        /**
         * The calendar quarter of the year.
         */
        CALENDAR_QUARTER((LocalDate l) -> l.get(IsoFields.QUARTER_OF_YEAR)),
        /**
         * The day of the quarter.
         */
        DAY_OF_QUARTER((LocalDate l) -> l.get(IsoFields.DAY_OF_QUARTER)),
        /**
         * The day.
         */
        DAY(LocalDate::getDayOfMonth),
        /**
         * The month.
         */
        MONTH(LocalDate::getMonthValue),
        /**
         * The year.
         */
        YEAR(LocalDate::getYear);

        private final ToIntFunction<LocalDate> extractionFunction;

        private DateFeatureType(ToIntFunction<LocalDate> func) {
            this.extractionFunction = func;
        }

        /**
         * Applies this enum's extraction function to the supplied date.
         * @param date The date to extract from.
         * @return The output of the extraction function.
         */
        public int extract(LocalDate date) {
            return extractionFunction.applyAsInt(date);
        }
    }

    @Config(mandatory = true, description = "The field name to read.")
    private String fieldName;

    @Config(mandatory = true, description = "The date features to extract.")
    private EnumSet<DateFeatureType> featureTypes;

    @Config(mandatory = true, description = "The expected date format.")
    private String dateFormat;
    private DateTimeFormatter formatter;

    @Config(mandatory = false, description = "Sets the locale language.")
    private String localeLanguage = "en";

    @Config(mandatory = false, description = "Sets the locale country.")
    private String localeCountry = "US";

    /**
     * For olcut.
     */
    private DateFieldProcessor() {}

    /**
     * Constructs a field processor which parses a date from the specified field name using the supplied
     * format string then extracts date features according to the supplied {@link EnumSet}.
     * <p>
     * Defaults to {@link Locale#US}.
     * <p>
     * Throws {@link IllegalArgumentException} if the date format is invalid.
     * @param fieldName The field name to read.
     * @param featureTypes The features to extract.
     * @param dateFormat The date format to use.
     */
    public DateFieldProcessor(String fieldName, EnumSet<DateFeatureType> featureTypes, String dateFormat) {
        this(fieldName,featureTypes,dateFormat,"en","US");
    }


    /**
     * Constructs a field processor which parses a date from the specified field name using the supplied
     * format string then extracts date features according to the supplied {@link EnumSet}.
     * <p>
     * Throws {@link IllegalArgumentException} if the date format is invalid.
     * @param fieldName The field name to read.
     * @param featureTypes The features to extract.
     * @param dateFormat The date format to use.
     * @param localeLanguage The Locale language.
     * @param localeCountry The Locale country.
     */
    public DateFieldProcessor(String fieldName, EnumSet<DateFeatureType> featureTypes, String dateFormat,
                              String localeLanguage, String localeCountry) {
        this.fieldName = fieldName;
        this.featureTypes = EnumSet.copyOf(featureTypes);
        this.dateFormat = dateFormat;
        this.localeLanguage = localeLanguage;
        this.localeCountry = localeCountry;
        postConfig();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        Locale locale = new Locale(localeLanguage, localeCountry);
        try {
            formatter = DateTimeFormatter.ofPattern(dateFormat, locale);
        } catch (IllegalArgumentException e) {
            throw new PropertyException(e,"","dateFormat","dateFormat could not be parsed by DateTimeFormatter");
        }
    }

    @Override
    public String getFieldName() {
        return fieldName;
    }

    @Override
    public List<ColumnarFeature> process(String value) {
        try {
            LocalDate date = LocalDate.parse(value, formatter);
            List<ColumnarFeature> features = new ArrayList<>(featureTypes.size());
            for (DateFeatureType f : featureTypes) {
                int featureValue = f.extract(date);
                ColumnarFeature feature = new ColumnarFeature(fieldName,f.toString(),featureValue);
                features.add(feature);
            }
            return features;
        } catch (DateTimeParseException e) {
            logger.log(Level.WARNING, e.getParsedString());
            logger.log(Level.WARNING, String.format("Unable to parse date %s with formatter %s", value, formatter.toString()));
            return Collections.emptyList();
        }
    }

    @Override
    public GeneratedFeatureType getFeatureType() {
        return GeneratedFeatureType.INTEGER;
    }

    @Override
    public FieldProcessor copy(String newFieldName) {
        return new DateFieldProcessor(newFieldName,featureTypes,dateFormat);
    }

    @Override
    public String toString() {
        return "DateFieldProcessor(" +
                "fieldName='" + fieldName + '\'' +
                ", featureTypes=" + featureTypes +
                ", dateFormat='" + dateFormat + '\'' +
                ", formatter=" + formatter +
                ", localeLanguage='" + localeLanguage + '\'' +
                ", localeCountry='" + localeCountry + '\'' +
                ')';
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"FieldProcessor");
    }
}
