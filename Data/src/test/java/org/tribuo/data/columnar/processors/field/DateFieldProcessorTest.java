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

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.tribuo.data.columnar.ColumnarFeature;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.Month;
import java.time.format.DateTimeFormatter;
import java.time.temporal.WeekFields;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class DateFieldProcessorTest {

    @BeforeAll
    public static void setup() {
        // Turn off the logger during the tests
        Logger logger = Logger.getLogger(DateFieldProcessor.class.getName());
        logger.setLevel(Level.OFF);
    }

    @Test
    public void testInvalidBehaviour() {
        String notADateFormatString = "not-a-date-format-string";
        try {
            DateFieldProcessor proc = new DateFieldProcessor("test",
                    EnumSet.of(DateFieldProcessor.DateFeatureType.DAY),
                    notADateFormatString);
            fail("Should have thrown on failing to parse the date format string");
        } catch (PropertyException e) {
            // pass
        }

        String isoFormat = "uuuu-MM-dd";
        DateFieldProcessor proc = new DateFieldProcessor("test",
                EnumSet.of(DateFieldProcessor.DateFeatureType.DAY),
                isoFormat);

        List<ColumnarFeature> extractedFeatures = proc.process("definitely-not-a-date");
        assertTrue(extractedFeatures.isEmpty());
    }

    @Test
    public void testValidBehaviour() {
        String isoFormat = "uuuu-MM-dd";
        DateTimeFormatter isoFormatter = DateTimeFormatter.ofPattern(isoFormat, Locale.US);
        String isoInput = "1994-01-26";
        DateFieldProcessor isoProc = new DateFieldProcessor("test-iso",
                EnumSet.allOf(DateFieldProcessor.DateFeatureType.class), isoFormat);
        LocalDate isoDate = LocalDate.parse(isoInput, isoFormatter);
        List<ColumnarFeature> isoFeatures = isoProc.process(isoInput);
        assertEquals(DateFieldProcessor.DateFeatureType.values().length, isoFeatures.size());
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY", isoDate.getDayOfMonth())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY", 26)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY_OF_WEEK", isoDate.getDayOfWeek().getValue())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY_OF_WEEK", DayOfWeek.WEDNESDAY.getValue())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY_OF_YEAR", isoDate.getDayOfYear())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY_OF_YEAR", 26)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "WEEK_OF_YEAR", isoDate.get(WeekFields.ISO.weekOfWeekBasedYear()))));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "WEEK_OF_MONTH", 4)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "EVEN_OR_ODD_DAY", 0)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "EVEN_OR_ODD_WEEK", 0)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "EVEN_OR_ODD_MONTH", 1)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "EVEN_OR_ODD_YEAR", 0)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "CALENDAR_QUARTER", 1)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "DAY_OF_QUARTER", 26)));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "MONTH", isoDate.getMonthValue())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "MONTH", Month.JANUARY.getValue())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "YEAR", isoDate.getYear())));
        assertTrue(isoFeatures.contains(
                new ColumnarFeature("test-iso", "YEAR", 1994)));

        String usFormat = "MM-dd-uuuu";
        DateTimeFormatter usFormatter = DateTimeFormatter.ofPattern(usFormat, Locale.US);
        String usInput = "09-08-1966";
        DateFieldProcessor usProc = new DateFieldProcessor("test-us",
                EnumSet.allOf(DateFieldProcessor.DateFeatureType.class), usFormat);
        LocalDate usDate = LocalDate.parse(usInput, usFormatter);
        List<ColumnarFeature> usFeatures = usProc.process(usInput);
        assertEquals(DateFieldProcessor.DateFeatureType.values().length, usFeatures.size());
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY", usDate.getDayOfMonth())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY", 8)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY_OF_WEEK", usDate.getDayOfWeek().getValue())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY_OF_WEEK", DayOfWeek.THURSDAY.getValue())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY_OF_YEAR", usDate.getDayOfYear())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY_OF_YEAR", 251)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "WEEK_OF_YEAR", usDate.get(WeekFields.ISO.weekOfWeekBasedYear()))));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "WEEK_OF_MONTH", 2)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "EVEN_OR_ODD_DAY", 1)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "EVEN_OR_ODD_WEEK", 0)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "EVEN_OR_ODD_MONTH", 1)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "EVEN_OR_ODD_YEAR", 0)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "CALENDAR_QUARTER", 3)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "DAY_OF_QUARTER", 70)));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "MONTH", usDate.getMonthValue())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "MONTH", Month.SEPTEMBER.getValue())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "YEAR", usDate.getYear())));
        assertTrue(usFeatures.contains(
                new ColumnarFeature("test-us", "YEAR", 1966)));


        String ukFormat = "dd-MM-uuuu";
        DateTimeFormatter ukFormatter = DateTimeFormatter.ofPattern(ukFormat, Locale.US);
        String ukInput = "23-11-1963";
        DateFieldProcessor ukProc = new DateFieldProcessor("test-uk",
                EnumSet.allOf(DateFieldProcessor.DateFeatureType.class), ukFormat);
        LocalDate ukDate = LocalDate.parse(ukInput, ukFormatter);
        List<ColumnarFeature> ukFeatures = ukProc.process(ukInput);
        assertEquals(DateFieldProcessor.DateFeatureType.values().length, ukFeatures.size());
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY", ukDate.getDayOfMonth())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY", 23)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY_OF_WEEK", ukDate.getDayOfWeek().getValue())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY_OF_WEEK", DayOfWeek.SATURDAY.getValue())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY_OF_YEAR", ukDate.getDayOfYear())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY_OF_YEAR", 327)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "WEEK_OF_YEAR", ukDate.get(WeekFields.ISO.weekOfWeekBasedYear()))));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "WEEK_OF_MONTH", 3)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "EVEN_OR_ODD_DAY", 1)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "EVEN_OR_ODD_WEEK", 1)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "EVEN_OR_ODD_MONTH", 1)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "EVEN_OR_ODD_YEAR", 1)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "CALENDAR_QUARTER", 4)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY_OF_QUARTER", 54)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "MONTH", ukDate.getMonthValue())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "MONTH", Month.NOVEMBER.getValue())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "YEAR", ukDate.getYear())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "YEAR", 1963)));
        ukProc = new DateFieldProcessor("test-uk",
                EnumSet.of(DateFieldProcessor.DateFeatureType.DAY,
                        DateFieldProcessor.DateFeatureType.MONTH,
                        DateFieldProcessor.DateFeatureType.YEAR),
                ukFormat);
        ukFeatures = ukProc.process(ukInput);
        assertEquals(3, ukFeatures.size());
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "DAY", 23)));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "MONTH", Month.NOVEMBER.getValue())));
        assertTrue(ukFeatures.contains(
                new ColumnarFeature("test-uk", "YEAR", 1963)));
    }

}
