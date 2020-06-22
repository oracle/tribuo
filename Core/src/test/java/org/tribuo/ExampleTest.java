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

package org.tribuo;

import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.ListExample;
import org.tribuo.test.MockOutput;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 *
 */
public class ExampleTest {

    private static final String[] featureNames = new String[]{"F0","F1","F2","F3","F4","F5","F6","F7","F8","F9"};

    @Test
    public void testArrayExampleDensify() {
        MockOutput output = new MockOutput("UNK");
        Example<MockOutput> example, expected;

        // Single feature
        example = new ArrayExample<>(output, new String[]{"F0"}, new double[]{1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0});
        checkDenseExample(expected,example);

        // Already dense
        example = new ArrayExample<>(output, featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        checkDenseExample(expected,example);

        // No edges
        example = new ArrayExample<>(output, new String[]{"F1","F3","F5","F6","F8"}, new double[]{1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0});
        checkDenseExample(expected,example);

        // Only edges
        example = new ArrayExample<>(output, new String[]{"F0","F1","F8","F9"}, new double[]{1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0});
        checkDenseExample(expected,example);
    }

    @Test
    public void testListExampleDensify() {
        MockOutput output = new MockOutput("UNK");
        Example<MockOutput> example, expected;

        // Single feature
        example = new ListExample<>(output, new String[]{"F0"}, new double[]{1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0});
        checkDenseExample(expected,example);

        // Already dense
        example = new ListExample<>(output, featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        checkDenseExample(expected,example);

        // No edges
        example = new ListExample<>(output, new String[]{"F1","F3","F5","F6","F8"}, new double[]{1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0});
        checkDenseExample(expected,example);

        // Only edges
        example = new ListExample<>(output, new String[]{"F0","F1","F8","F9"}, new double[]{1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0});
        checkDenseExample(expected,example);

    }

    public static void checkDenseExample(Example<MockOutput> expected, Example<MockOutput> actual) {
        assertEquals(expected.size(),actual.size());
        Iterator<Feature> expectedItr = expected.iterator();
        Iterator<Feature> actualItr = actual.iterator();
        while (expectedItr.hasNext() && actualItr.hasNext()) {
            Feature ef = expectedItr.next();
            Feature af = actualItr.next();
            assertEquals(ef.getName(),af.getName());
            assertEquals(ef.getValue(),af.getValue(),1e-12);
        }
        assertEquals(expected, actual);
    }

    @Test
    public void metadataTest() {
        MockOutput output = new MockOutput("UNK");

        ArrayExample<MockOutput> test;

        test = new ArrayExample<>(output, Collections.emptyMap());
        // Check that it starts empty
        assertTrue(test.getMetadata().isEmpty());
        assertFalse(test.getMetadataValue("Bananas").isPresent());
        // Check that appends work
        test.setMetadataValue("Bananas","Yellow");
        assertTrue(test.containsMetadata("Bananas"));
        assertEquals(test.getMetadataValue("Bananas").get(),"Yellow");
        assertEquals(1,test.getMetadata().size());
        // Check that the metadata grows appropriately
        test.setMetadataValue("Oranges","Orange");
        assertEquals(test.getMetadataValue("Oranges").get(),"Orange");
        Map<String,Object> metadata = test.getMetadata();
        assertEquals(2,metadata.size());
        // Check that the metadata returned is a copy
        metadata.put("Strawberries","Red");
        assertEquals(2,test.getMetadata().size());
        // Check that overwriting throws
        assertThrows(IllegalArgumentException.class,() -> test.setMetadataValue("Bananas","Pink"));
        assertEquals(2,test.getMetadata().size());
        assertEquals(test.getMetadataValue("Bananas").get(),"Yellow");
        assertEquals(test.getMetadataValue("Oranges").get(),"Orange");
        // Check that the metadata is copied
        ArrayExample<MockOutput> copy = test.copy();
        assertEquals(2,copy.getMetadata().size());
        assertEquals(copy.getMetadataValue("Bananas").get(),"Yellow");
        assertEquals(copy.getMetadataValue("Oranges").get(),"Orange");
        // Check that the copies are independent
        copy.setMetadataValue("Strawberries","Red");
        assertEquals(3,copy.getMetadata().size());
        assertEquals(2,test.getMetadata().size());
    }

    @Test
    public void invalidArrayExampleTest() {
        MockOutput output = new MockOutput("UNK");
        ArrayExample<MockOutput> test = new ArrayExample<>(output);

        // Empty examples are invalid.
        assertFalse(test.validateExample());

        test.add(new Feature("test",1.0));
        test.add(new Feature("test",1.0));

        // Examples with duplicate features are invalid
        assertFalse(test.validateExample());

        test = new ArrayExample<>(output);

        test.add(new Feature("test",Double.NaN));

        // Examples with NaN valued features are invalid
        assertFalse(test.validateExample());

        test = new ArrayExample<>(output);

        test.add(new Feature("test-1",1.0));
        test.add(new Feature("test-2",1.0));

        // This example should be valid
        assertTrue(test.validateExample());
    }

    @Test
    public void invalidListExampleTest() {
        MockOutput output = new MockOutput("UNK");
        ListExample<MockOutput> test = new ListExample<>(output);

        // Empty examples are invalid.
        assertFalse(test.validateExample());

        test.add(new Feature("test",1.0));
        test.add(new Feature("test",1.0));

        // Examples with duplicate features are invalid
        assertFalse(test.validateExample());

        test = new ListExample<>(output);

        test.add(new Feature("test",Double.NaN));

        // Examples with NaN valued features are invalid
        assertFalse(test.validateExample());

        test = new ListExample<>(output);

        test.add(new Feature("test-1",1.0));
        test.add(new Feature("test-2",1.0));

        // This example should be valid
        assertTrue(test.validateExample());
    }
}
