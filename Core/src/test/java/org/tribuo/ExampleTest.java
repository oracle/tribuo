/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.impl.BinaryFeaturesExample;
import org.tribuo.impl.ListExample;
import org.tribuo.test.MockOutput;
import org.tribuo.util.Merger;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.tribuo.test.Helpers.testProtoSerialization;

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
        testProtoSerialization(example);

        // Already dense
        example = new ArrayExample<>(output, featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        checkDenseExample(expected,example);
        testProtoSerialization(example);

        // No edges
        example = new ArrayExample<>(output, new String[]{"F1","F3","F5","F6","F8"}, new double[]{1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0});
        checkDenseExample(expected,example);
        testProtoSerialization(example);

        // Only edges
        example = new ArrayExample<>(output, new String[]{"F0","F1","F8","F9"}, new double[]{1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ArrayExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0});
        checkDenseExample(expected,example);
        testProtoSerialization(example);
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
        testProtoSerialization(example);

        // Already dense
        example = new ListExample<>(output, featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0});
        checkDenseExample(expected,example);
        testProtoSerialization(example);

        // No edges
        example = new ListExample<>(output, new String[]{"F1","F3","F5","F6","F8"}, new double[]{1.0,1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{0.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0});
        checkDenseExample(expected,example);
        testProtoSerialization(example);

        // Only edges
        example = new ListExample<>(output, new String[]{"F0","F1","F8","F9"}, new double[]{1.0,1.0,1.0,1.0});
        example.densify(Arrays.asList(featureNames));
        expected = new ListExample<>(new MockOutput("UNK"), featureNames, new double[]{1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0});
        checkDenseExample(expected,example);
        testProtoSerialization(example);
    }

    @Test
    public void testArrayExampleRemove() {
        String[] names = new String[]{"A","B","C","D","E"};
        double[] values = new double[]{1.0,1.0,1.0,1.0,1.0};
        MockOutput output = new MockOutput("UNK");

        List<Feature> featureList;
        Example<MockOutput> example;

        example = new ArrayExample<>(output,names,values);
        example.removeFeatures(Collections.singletonList(new Feature("E",1.0)));
        assertEquals(4,example.size());
        testProtoSerialization(example);

        example = new ArrayExample<>(output,names,values);
        featureList = new ArrayList<>();
        featureList.add(new Feature("D",1.0));
        featureList.add(new Feature("C",1.0));
        featureList.add(new Feature("alpha",1.0));
        example.removeFeatures(featureList);
        assertEquals(3,example.size());
        assertEquals("A",example.lookup("A").name);
        assertEquals("B",example.lookup("B").name);
        assertEquals("E",example.lookup("E").name);
        testProtoSerialization(example);

        example = new ArrayExample<>(output,names,values);
        featureList = new ArrayList<>();
        featureList.add(new Feature("D",1.0));
        featureList.add(new Feature("D",1.0));
        featureList.add(new Feature("B",1.0));
        featureList.add(new Feature("E",1.0));
        example.removeFeatures(featureList);
        assertEquals(2,example.size());
        assertEquals("A",example.lookup("A").name);
        assertEquals("C",example.lookup("C").name);

        example = new ArrayExample<>(output,new String[]{"A","B","C","D","E","A","C","E"},new double[]{1,1,1,1,1,1,1,1});
        featureList = new ArrayList<>();
        featureList.add(new Feature("D",1.0));
        featureList.add(new Feature("D",1.0));
        featureList.add(new Feature("B",1.0));
        featureList.add(new Feature("E",1.0));
        example.removeFeatures(featureList);
        assertEquals(4,example.size());
        assertEquals("A",example.lookup("A").name);
        assertEquals("C",example.lookup("C").name);
        testProtoSerialization(example);
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

    @Test
    public void exampleIterators() {
        MockOutput output = new MockOutput("UNK");
        List<Feature> features = new ArrayList<>();
        features.add(new Feature("A",1.0));
        features.add(new Feature("C",1.0));
        features.add(new Feature("B",1.0));

        ArrayExample<MockOutput> array = new ArrayExample<>(output,features);
        assertEquals(3,array.size());

        Iterator<Feature> arrayItr = array.iterator();
        assertTrue(arrayItr.hasNext());
        assertEquals(features.get(0),arrayItr.next());
        assertTrue(arrayItr.hasNext());
        // Features are lexicographically sorted inside examples
        assertEquals(features.get(2),arrayItr.next());
        assertTrue(arrayItr.hasNext());
        assertEquals(features.get(1),arrayItr.next());
        assertFalse(arrayItr.hasNext());
        assertThrows(NoSuchElementException.class, arrayItr::next);

        ListExample<MockOutput> list = new ListExample<>(output,features);
        assertEquals(3,list.size());

        Iterator<Feature> listItr = list.iterator();
        assertTrue(listItr.hasNext());
        assertEquals(features.get(0),listItr.next());
        assertTrue(listItr.hasNext());
        // Features are lexicographically sorted inside examples
        assertEquals(features.get(2),listItr.next());
        assertTrue(listItr.hasNext());
        assertEquals(features.get(1),listItr.next());
        assertFalse(listItr.hasNext());
        assertThrows(NoSuchElementException.class, listItr::next);

        BinaryFeaturesExample<MockOutput> binary = new BinaryFeaturesExample<>(output,features);
        assertEquals(3,binary.size());

        Iterator<Feature> binaryItr = binary.iterator();
        assertTrue(binaryItr.hasNext());
        assertEquals(features.get(0),binaryItr.next());
        assertTrue(binaryItr.hasNext());
        // Features are lexicographically sorted inside examples
        assertEquals(features.get(2),binaryItr.next());
        assertTrue(binaryItr.hasNext());
        assertEquals(features.get(1),binaryItr.next());
        assertFalse(binaryItr.hasNext());
        assertThrows(NoSuchElementException.class, binaryItr::next);
    }
    
    @Test
    public void testBinaryFeaturesExample() {
        MockOutput output = new MockOutput("UNK");
        Example<MockOutput> test = new BinaryFeaturesExample<>(output);
        // Empty examples are invalid.
        assertFalse(test.validateExample());
        testProtoSerialization(test);

        test.add(new Feature("test",1.0));
        test.add(new Feature("test",1.0));
        assertFalse(test.validateExample());
        test.reduceByName(Merger.max());
        assertTrue(test.validateExample());
        //try adding a non-binary feature
        Assertions.assertThrows(IllegalArgumentException.class, () -> test.add(new Feature("test-2",2.0)));
        Assertions.assertThrows(UnsupportedOperationException.class, () -> test.transform(null));
        
        test.add(new Feature("test-2",1.0));
        testProtoSerialization(test);

        Example<MockOutput> test2 = test.copy();
        test2.add(new Feature("test-2",1.0));
        test2.reduceByName(Merger.max());
        assertTrue(test.validateExample());

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
        Example<MockOutput> copy = test.copy();
        assertEquals(2,copy.getMetadata().size());
        assertEquals(copy.getMetadataValue("Bananas").get(),"Yellow");
        assertEquals(copy.getMetadataValue("Oranges").get(),"Orange");
        // Check that the copies are independent
        copy.setMetadataValue("Strawberries","Red");
        assertEquals(3,copy.getMetadata().size());
        assertEquals(2,test.getMetadata().size());
        testProtoSerialization(test);

        Feature lookup = test.lookup("test-2");
        assertEquals("test-2", lookup.name);
        assertEquals(1.0, lookup.value);
        assertTrue(BinaryFeaturesExample.isBinary(lookup));
        ((BinaryFeaturesExample<MockOutput>) test).add("f3");
        assertThrows(IllegalArgumentException.class, () -> BinaryFeaturesExample.checkIsBinary(new Feature("f4", 2.0)));
        
        int count = 0;
        for(Feature feature : test) {
            assertTrue(BinaryFeaturesExample.isBinary(feature));
            count++;
        }
        assertEquals(3, count);
    }
}
