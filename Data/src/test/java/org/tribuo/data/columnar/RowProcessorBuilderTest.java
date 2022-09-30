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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.Test;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.data.columnar.extractors.DateExtractor;
import org.tribuo.data.columnar.extractors.FloatExtractor;
import org.tribuo.data.columnar.extractors.IdentityExtractor;
import org.tribuo.data.columnar.extractors.IntExtractor;
import org.tribuo.data.columnar.extractors.OffsetDateTimeExtractor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.data.columnar.processors.field.TextFieldProcessor;
import org.tribuo.data.text.impl.TokenPipeline;
import org.tribuo.test.MockOutput;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.time.LocalDate;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class RowProcessorBuilderTest {

    @Test
    public void testInvalidRegexMapping() {
        List<String> fieldNames = Arrays.asList("Armadillos", "Armadas", "Archery", "Battleship", "Battles", "Carrots", "Label");

        Map<String, FieldProcessor> fixed = new HashMap<>();

        fixed.put("Battles", new IdentityProcessor("Battles"));

        Map<String, FieldProcessor> regex = new HashMap<>();

        try {
            regex.put("Arma*", new IdentityProcessor("Arma*"));
            regex.put("Monkeys", new IdentityProcessor("Monkeys"));
            RowProcessor<MockOutput> rowProcessor = new RowProcessor.Builder<MockOutput>()
                    .setRegexMappingProcessors(regex)
                    .setFieldProcessors(fixed.values())
                    .build(new MockResponseProcessor("Label"));
            rowProcessor.expandRegexMapping(fieldNames);
            fail("Should have thrown an IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // pass
        } catch (Exception e) {
            fail("Incorrect exception thrown.: " + e.getMessage());
        }

        regex.clear();

        try {
            regex.put("Battle*", new IdentityProcessor("Battle*"));
            RowProcessor<MockOutput> rowProcessor = new RowProcessor.Builder<MockOutput>()
                    .setRegexMappingProcessors(regex)
                    .setFieldProcessors(fixed.values())
                    .build(new MockResponseProcessor("Label"));
            rowProcessor.expandRegexMapping(fieldNames);
            fail("Should have thrown an IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // pass
        } catch (Exception e) {
            fail("Incorrect exception thrown.");
        }

        regex.clear();

        try {
            regex.put("Arm*", new IdentityProcessor("Arm*"));
            regex.put("Armadil*", new IdentityProcessor("Armadil*"));
            RowProcessor<MockOutput> rowProcessor = new RowProcessor.Builder<MockOutput>()
                    .setRegexMappingProcessors(regex)
                    .setFieldProcessors(fixed.values())
                    .build(new MockResponseProcessor("Label"));
            rowProcessor.expandRegexMapping(fieldNames);
            fail("Should have thrown an IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // pass
        } catch (Exception e) {
            fail("Incorrect exception thrown.");
        }
    }

    @Test
    public void metadataExtractorTest() {
        Map<String, FieldProcessor> fixed = new HashMap<>();

        fixed.put("Battles", new IdentityProcessor("Battles"));
        fixed.put("Armadas", new DoubleFieldProcessor("Armadas"));

        List<FieldExtractor<?>> metadataExtractors = new ArrayList<>();
        metadataExtractors.add(new IdentityExtractor("Armadillos", Example.NAME));
        metadataExtractors.add(new IntExtractor("Armadillos", "ID"));
        metadataExtractors.add(new DateExtractor("Carrots", "Date", "uuuuMMdd"));
        metadataExtractors.add(new OffsetDateTimeExtractor("Carrot-time", "OffsetDateTime", "dd/MM/yyyy HH:mmx"));

        FloatExtractor weightExtractor = new FloatExtractor("Mass");

        MockResponseProcessor response = new MockResponseProcessor("Label");

        Map<String, String> row = new HashMap<>();
        row.put("Armadillos", "1");
        row.put("Armadas", "2");
        row.put("Archery", "3");
        row.put("Battleship", "4");
        row.put("Battles", "5");
        row.put("Carrots", "20010506");
        row.put("Carrot-time", "14/10/2020 16:07+01");
        row.put("Mass", "9000");
        row.put("Label", "Sheep");

        RowProcessor<MockOutput> processor = new RowProcessor.Builder<MockOutput>()
                .setMetadataExtractors(metadataExtractors)
                .setWeightExtractor(weightExtractor)
                .setFieldProcessors(fixed.values())
                .build(response);

        Example<MockOutput> example = processor.generateExample(row, true).get();

        // Check example is extracted correctly
        assertEquals(2, example.size());
        assertEquals("Sheep", example.getOutput().label);
        Iterator<Feature> featureIterator = example.iterator();
        Feature a = featureIterator.next();
        assertEquals("Armadas@value", a.getName());
        assertEquals(2.0, a.getValue());
        a = featureIterator.next();
        assertEquals("Battles@5", a.getName());
        assertEquals(IdentityProcessor.FEATURE_VALUE, a.getValue());
        assertEquals(9000f, example.getWeight());

        // Check metadata is extracted correctly
        Map<String, Object> metadata = example.getMetadata();
        assertEquals(4, metadata.size());
        assertEquals("1", metadata.get(Example.NAME));
        assertEquals(1, metadata.get("ID"));
        assertEquals(LocalDate.of(2001, 5, 6), metadata.get("Date"));
        assertEquals(OffsetDateTime.of(LocalDate.of(2020, 10, 14), LocalTime.of(16, 7), ZoneOffset.ofHours(1)), metadata.get("OffsetDateTime"));

        // Check metadata types
        Map<String, Class<?>> metadataTypes = processor.getMetadataTypes();
        assertEquals(4, metadataTypes.size());
        assertEquals(String.class, metadataTypes.get(Example.NAME));
        assertEquals(Integer.class, metadataTypes.get("ID"));
        assertEquals(LocalDate.class, metadataTypes.get("Date"));
        assertEquals(OffsetDateTime.class, metadataTypes.get("OffsetDateTime"));

        // Check an invalid metadata extractor throws IllegalArgumentException
        List<FieldExtractor<?>> badExtractors = new ArrayList<>();
        badExtractors.add(new IdentityExtractor("Armadillos", Example.NAME));
        badExtractors.add(new IntExtractor("Armadillos", "ID"));
        badExtractors.add(new DateExtractor("Carrots", "ID", "uuuuMMdd"));

        assertThrows(PropertyException.class, () -> new RowProcessor<>(badExtractors, weightExtractor, response, fixed, Collections.emptySet()));
    }

    @Test
    public void replaceNewlinesWithSpacesTest() {
        final Pattern BLANK_LINES = Pattern.compile("(\n[\\s-]*\n)+");

        final Function<CharSequence, CharSequence> newLiner = (CharSequence charSequence) -> {
            if (charSequence == null || charSequence.length() == 0) {
                return charSequence;
            }
            return BLANK_LINES.splitAsStream(charSequence).collect(Collectors.joining(" *\n\n"));
        };

        Tokenizer tokenizer = new RowProcessorTest.MungingTokenizer(new BreakIteratorTokenizer(Locale.US), newLiner);
        TokenPipeline textPipeline = new TokenPipeline(tokenizer, 2, false);

        final Map<String, FieldProcessor> fieldProcessors = new HashMap<>();
        fieldProcessors.put("order_text", new TextFieldProcessor("order_text", textPipeline));

        MockResponseProcessor response = new MockResponseProcessor("Label");

        Map<String, String> row = new HashMap<>();
        row.put("order_text", "Jimmy\n\n\n\nHoffa");
        row.put("Label", "Sheep");

        RowProcessor<MockOutput> processor = new RowProcessor.Builder<MockOutput>()
                .setReplaceNewLinesWithSpaces(false)
                .setFieldProcessors(fieldProcessors.values())
                .build(response);

        Example<MockOutput> example = processor.generateExample(row, true).get();

        // Check example is extracted correctly
        assertEquals(5, example.size());
        assertEquals("Sheep", example.getOutput().label);
        Iterator<Feature> featureIterator = example.iterator();
        Feature a = featureIterator.next();
        assertEquals("order_text@1-N=*", a.getName());
        assertEquals(1.0, a.getValue());
        a = featureIterator.next();
        assertEquals("order_text@1-N=Hoffa", a.getName());
        a = featureIterator.next();
        assertEquals("order_text@1-N=Jimmy", a.getName());
        a = featureIterator.next();
        assertEquals("order_text@2-N=*/Hoffa", a.getName());
        a = featureIterator.next();
        assertEquals("order_text@2-N=Jimmy/*", a.getName());
        assertFalse(featureIterator.hasNext());

        // same input with replaceNewlinesWithSpacesTest=true (the default) produces different features
        processor = new RowProcessor.Builder<MockOutput>()
                .setReplaceNewLinesWithSpaces(true)
                .setFieldProcessors(fieldProcessors.values())
                .build(response);

        example = processor.generateExample(row, true).get();

        // Check example is extracted correctly
        assertEquals(3, example.size());
        assertEquals("Sheep", example.getOutput().label);
        featureIterator = example.iterator();
        a = featureIterator.next();
        assertEquals("order_text@1-N=Hoffa", a.getName());
        assertEquals(1.0, a.getValue());
        a = featureIterator.next();
        assertEquals("order_text@1-N=Jimmy", a.getName());
        a = featureIterator.next();
        assertEquals("order_text@2-N=Jimmy/Hoffa", a.getName());
        assertFalse(featureIterator.hasNext());
    }
}
