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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.data.columnar.extractors.DateExtractor;
import org.tribuo.data.columnar.extractors.FloatExtractor;
import org.tribuo.data.columnar.extractors.IdentityExtractor;
import org.tribuo.data.columnar.extractors.IntExtractor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.field.IdentityProcessor;
import org.tribuo.test.MockOutput;
import org.junit.jupiter.api.Test;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class RowProcessorTest {

    @Test
    public void testInvalidRegexMapping() {
        List<String> fieldNames = Arrays.asList("Armadillos", "Armadas", "Archery", "Battleship", "Battles", "Carrots", "Label");

        Map<String, FieldProcessor> fixed = new HashMap<>();

        fixed.put("Battles", new IdentityProcessor("Battles"));

        Map<String, FieldProcessor> regex = new HashMap<>();

        try {
            regex.put("Arma*", new IdentityProcessor("Arma*"));
            regex.put("Monkeys", new IdentityProcessor("Monkeys"));
            RowProcessor<MockOutput> rowProcessor = new RowProcessor<>(Collections.emptyList(), null, new MockResponseProcessor("Label"), fixed, regex, new HashSet<>());
            rowProcessor.expandRegexMapping(fieldNames);
            fail("Should have thrown an IllegalArgumentException");
        } catch (IllegalArgumentException e) {
            // pass
        } catch (Exception e) {
            fail("Incorrect exception thrown.");
        }

        regex.clear();

        try {
            regex.put("Battle*", new IdentityProcessor("Battle*"));
            RowProcessor<MockOutput> rowProcessor = new RowProcessor<>(Collections.emptyList(), null, new MockResponseProcessor("Label"), fixed, regex, new HashSet<>());
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
            RowProcessor<MockOutput> rowProcessor = new RowProcessor<>(Collections.emptyList(), null, new MockResponseProcessor("Label"), fixed, regex, new HashSet<>());
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
        List<String> fieldNames = Arrays.asList("Armadillos", "Armadas", "Archery", "Battleship", "Battles", "Carrots", "Mass", "Label");

        Map<String, FieldProcessor> fixed = new HashMap<>();

        fixed.put("Battles", new IdentityProcessor("Battles"));
        fixed.put("Armadas", new DoubleFieldProcessor("Armadas"));

        List<FieldExtractor<?>> metadataExtractors = new ArrayList<>();
        metadataExtractors.add(new IdentityExtractor("Armadillos", Example.NAME));
        metadataExtractors.add(new IntExtractor("Armadillos", "ID"));
        metadataExtractors.add(new DateExtractor("Carrots","Date",DateTimeFormatter.BASIC_ISO_DATE));

        FloatExtractor weightExtractor = new FloatExtractor("Mass");

        MockResponseProcessor response = new MockResponseProcessor("Label");

        Map<String,String> row = new HashMap<>();
        row.put("Armadillos","1");
        row.put("Armadas","2");
        row.put("Archery","3");
        row.put("Battleship","4");
        row.put("Battles","5");
        row.put("Carrots","20010506");
        row.put("Mass","9000");
        row.put("Label","Sheep");

        RowProcessor<MockOutput> processor = new RowProcessor<>(metadataExtractors,weightExtractor,response,fixed,Collections.emptySet());

        Example<MockOutput> example = processor.generateExample(row,true).get();

        // Check example is extracted correctly
        assertEquals(2,example.size());
        assertEquals("Sheep",example.getOutput().label);
        Iterator<Feature> featureIterator = example.iterator();
        Feature a = featureIterator.next();
        assertEquals("Armadas@value", a.getName());
        assertEquals(2.0, a.getValue());
        a = featureIterator.next();
        assertEquals("Battles@5", a.getName());
        assertEquals(IdentityProcessor.FEATURE_VALUE,a.getValue());
        assertEquals(9000f,example.getWeight());

        // Check metadata is extracted correctly
        Map<String,Object> metadata = example.getMetadata();
        assertEquals(3,metadata.size());
        assertEquals("1",metadata.get(Example.NAME));
        assertEquals(1,metadata.get("ID"));
        assertEquals(LocalDate.of(2001,5,6),metadata.get("Date"));

        // Check metadata types
        Map<String,Class<?>> metadataTypes = processor.getMetadataTypes();
        assertEquals(3,metadataTypes.size());
        assertEquals(String.class,metadataTypes.get(Example.NAME));
        assertEquals(Integer.class,metadataTypes.get("ID"));
        assertEquals(LocalDate.class,metadataTypes.get("Date"));

        // Check an invalid metadata extractor throws IllegalArgumentException
        List<FieldExtractor<?>> badExtractors = new ArrayList<>();
        badExtractors.add(new IdentityExtractor("Armadillos", Example.NAME));
        badExtractors.add(new IntExtractor("Armadillos", "ID"));
        badExtractors.add(new DateExtractor("Carrots", "ID",DateTimeFormatter.BASIC_ISO_DATE));

        assertThrows(PropertyException.class, () -> new RowProcessor<>(badExtractors,weightExtractor,response,fixed,Collections.emptySet()));
    }

}
