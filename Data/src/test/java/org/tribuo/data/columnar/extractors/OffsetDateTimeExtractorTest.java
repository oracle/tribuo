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

package org.tribuo.data.columnar.extractors;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class OffsetDateTimeExtractorTest {

    @BeforeAll
    public static void setup() {
        // Turn off the logger during the tests
        Logger logger = Logger.getLogger(OffsetDateTimeExtractor.class.getName());
        logger.setLevel(Level.OFF);
    }

    @Test
    public void testInvalidBehaviour() {
        String notADateFormatString = "not-a-date-format-string";
        try {
            OffsetDateTimeExtractor extractor = new OffsetDateTimeExtractor("test","date", notADateFormatString);
            fail("Should have thrown on failing to parse the date format string");
        } catch (PropertyException e) {
            // pass
        }

        String isoFormat = "uuuu-MM-dd";
        OffsetDateTimeExtractor extractor = new OffsetDateTimeExtractor("test", "date", isoFormat);

        Optional<OffsetDateTime> extractedMetadata = extractor.extractField("definitely-not-a-date");
        assertFalse(extractedMetadata.isPresent());
    }

    @Test
    public void testValidBehaviour() {
        String isoFormat = "uuuu-MM-dd HH:mmx";
        DateTimeFormatter isoFormatter = DateTimeFormatter.ofPattern(isoFormat, Locale.US);
        String isoInput = "1994-01-26 20:00+05";
        OffsetDateTimeExtractor isoExtractor = new OffsetDateTimeExtractor("test-iso", "date-iso", isoFormat);
        OffsetDateTime isoDate = OffsetDateTime.parse(isoInput, isoFormatter);
        Optional<OffsetDateTime> isoExtracted = isoExtractor.extractField(isoInput);
        assertTrue(isoExtracted.isPresent());
        assertEquals(isoDate,isoExtracted.get());

        String usFormat = "MM-dd-uuuu HH:mmx";
        DateTimeFormatter usFormatter = DateTimeFormatter.ofPattern(usFormat, Locale.US);
        String usInput = "09-08-1966 20:30+05";
        OffsetDateTimeExtractor usExtractor = new OffsetDateTimeExtractor("test-us", "date-us", usFormat);
        OffsetDateTime usDate = OffsetDateTime.parse(usInput, usFormatter);
        Optional<OffsetDateTime> usExtracted = usExtractor.extractField(usInput);
        assertTrue(usExtracted.isPresent());
        assertEquals(usDate,usExtracted.get());

        String ukFormat = "dd-MM-uuuu HH:mmx";
        DateTimeFormatter ukFormatter = DateTimeFormatter.ofPattern(ukFormat, Locale.US);
        String ukInput = "23-11-1963 17:16+00";
        OffsetDateTimeExtractor ukProc = new OffsetDateTimeExtractor("test-uk", "date-uk", ukFormat);
        OffsetDateTime ukDate = OffsetDateTime.parse(ukInput, ukFormatter);
        Optional<OffsetDateTime> ukExtracted = ukProc.extractField(ukInput);
        assertTrue(ukExtracted.isPresent());
        assertEquals(ukDate,ukExtracted.get());
    }

}
