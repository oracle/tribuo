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
import org.tribuo.test.MockDataSourceProvenance;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;
import org.tribuo.util.Merger;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

public class FeatureTest {

    private static final Logger logger = Logger.getLogger(FeatureTest.class.getName());

    @Test
    public void testFeatureConflictingIDs() {
        Example<MockOutput> ex = new ListExample<>(new MockOutput("0"));
        List<Feature> fs = new ArrayList<>();
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#0", 1));
        ex.addAll(fs);
        MutableDataset<MockOutput> data = new MutableDataset<>(new MockDataSourceProvenance(), new MockOutputFactory());
        try {
            data.add(ex);
            fail("should have thrown exception for duplicate features");
        } catch (IllegalArgumentException e) {
            //pass
        }
    }

    @Test
    public void testFeatureMerging() {
        // List example
        Example<MockOutput> example = new ListExample<>(new MockOutput("0"));
        testShortMerge("ListExample",example);
        example = new ListExample<>(new MockOutput("0"));
        testLongMerge("ListExample",example);
        example = new ListExample<>(new MockOutput("0"));
        testDifferentLongMerge("ListExample",example);
        example = new ListExample<>(new MockOutput("0"));
        testShortMergeWithNoRepeats("ListExample",example);
        example = new ListExample<>(new MockOutput("0"));
        testMergeWithNoRepeats("ListExample",example);
        example = new ListExample<>(new MockOutput("0"));
        testMergeOnEmptyExample("ListExample",example);

        // Array example
        example = new ArrayExample<>(new MockOutput("0"));
        testShortMerge("ArrayExample",example);
        example = new ArrayExample<>(new MockOutput("0"));
        testLongMerge("ArrayExample",example);
        example = new ArrayExample<>(new MockOutput("0"));
        testDifferentLongMerge("ArrayExample",example);
        example = new ArrayExample<>(new MockOutput("0"));
        testShortMergeWithNoRepeats("ArrayExample",example);
        example = new ArrayExample<>(new MockOutput("0"));
        testMergeWithNoRepeats("ArrayExample",example);
        example = new ArrayExample<>(new MockOutput("0"));
        testMergeOnEmptyExample("ArrayExample",example);
    }

    public void testShortMerge(String name, Example<MockOutput> ex) {
        List<Feature> fs = new ArrayList<>();
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#0", 1));
        ex.addAll(fs);
        ex.reduceByName(Merger.add());
        assertEquals(2,ex.size(), name+": Features not merged correctly");
        assertEquals(2.0,ex.lookup("F#0").getValue(),1e-5, name+": Feature 0 should have a count of 2");
    }

    public void testLongMerge(String name, Example<MockOutput> ex) {
        List<Feature> fs = new ArrayList<>();
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#2", 1));
        fs.add(new Feature("F#2", 1));
        fs.add(new Feature("F#0", 1));
        ex.addAll(fs);
        ex.reduceByName(Merger.add());
        assertEquals(3,ex.size(), name+": Features not merged correctly");
        assertEquals(5.0,ex.lookup("F#0").getValue(),1e-5, name+": Feature 0 should have a count of 5");
        assertEquals(2.0,ex.lookup("F#1").getValue(),1e-5, name+": Feature 1 should have a count of 2");
        assertEquals(2.0,ex.lookup("F#2").getValue(),1e-5, name+": Feature 2 should have a count of 2");
    }

    public void testDifferentLongMerge(String name, Example<MockOutput> ex) {
        List<Feature> fs = new ArrayList<>();
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#3", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#4", 1));
        fs.add(new Feature("F#2", 1));
        fs.add(new Feature("F#2", 1));
        fs.add(new Feature("F#5", 1));
        ex.addAll(fs);
        ex.reduceByName(Merger.add());
        assertEquals(6,ex.size(), name+": Features not merged correctly");
        assertEquals(2.0,ex.lookup("F#0").getValue(),1e-5, name+": Feature 0 should have a count of 2");
        assertEquals(2.0,ex.lookup("F#1").getValue(),1e-5, name+": Feature 1 should have a count of 2");
        assertEquals(2.0,ex.lookup("F#2").getValue(),1e-5, name+": Feature 2 should have a count of 2");
        assertEquals(1.0,ex.lookup("F#3").getValue(),1e-5, name+": Feature 3 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#4").getValue(),1e-5, name+": Feature 4 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#5").getValue(),1e-5, name+": Feature 5 should have a count of 1");
    }

    public void testShortMergeWithNoRepeats(String name, Example<MockOutput> ex) {
        List<Feature> fs = new ArrayList<>();
        fs.add(new Feature("F#0", 1));
        ex.addAll(fs);
        ex.reduceByName(Merger.add());
        assertEquals(1,ex.size(), name+": Features merged when they shouldn't have");
        assertEquals(1.0,ex.lookup("F#0").getValue(),1e-5, name+": Feature 0 should have a count of 1");
    }

    public void testMergeWithNoRepeats(String name, Example<MockOutput> ex) {
        List<Feature> fs = new ArrayList<>();
        fs.add(new Feature("F#0", 1));
        fs.add(new Feature("F#1", 1));
        fs.add(new Feature("F#2", 1));
        fs.add(new Feature("F#3", 1));
        fs.add(new Feature("F#4", 1));
        fs.add(new Feature("F#5", 1));
        fs.add(new Feature("F#6", 1));
        fs.add(new Feature("F#7", 1));
        fs.add(new Feature("F#8", 1));
        ex.addAll(fs);
        ex.reduceByName(Merger.add());
        assertEquals(9,ex.size(), name+": Features merged when they shouldn't have");
        assertEquals(1.0,ex.lookup("F#0").getValue(),1e-5, name+": Feature 0 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#1").getValue(),1e-5, name+": Feature 1 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#2").getValue(),1e-5, name+": Feature 2 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#3").getValue(),1e-5, name+": Feature 3 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#4").getValue(),1e-5, name+": Feature 4 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#5").getValue(),1e-5, name+": Feature 5 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#6").getValue(),1e-5, name+": Feature 6 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#7").getValue(),1e-5, name+": Feature 7 should have a count of 1");
        assertEquals(1.0,ex.lookup("F#8").getValue(),1e-5, name+": Feature 8 should have a count of 1");
    }

    public void testMergeOnEmptyExample(String name, Example<MockOutput> ex) {
        List<Feature> fs = new ArrayList<>();
        ex.addAll(fs);
        ex.reduceByName(Merger.add());
        assertEquals(0,ex.size(), name+": Merging an empty example created a feature");
    }
}
