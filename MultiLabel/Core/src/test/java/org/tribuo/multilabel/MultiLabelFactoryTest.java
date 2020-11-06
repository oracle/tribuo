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

package org.tribuo.multilabel;

import org.tribuo.classification.Label;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MultiLabelFactoryTest {

    @Test
    public void testGenerateOutput_str() {
        MultiLabelFactory factory = new MultiLabelFactory();
        MultiLabel output = factory.generateOutput("a=true,b=true,c=true");
        assertEquals(3, output.getLabelSet().size());
        assertEquals("a,b,c", output.getLabelString());

        output = factory.generateOutput("a,b,c");
        assertEquals(3, output.getLabelSet().size());
        assertEquals("a,b,c", output.getLabelString());

        output = factory.generateOutput("a,b");
        assertEquals(2, output.getLabelSet().size());
        assertEquals("a,b", output.getLabelString());

        output = factory.generateOutput("a");
        assertEquals(1, output.getLabelSet().size());
        assertEquals("a", output.getLabelString());

        //
        // Boolean.parseBoolean("integer") resolves to false.
        output = factory.generateOutput("a=1,b=1,c=0");
        assertEquals(0, output.getLabelSet().size());
        assertEquals("", output.getLabelString());
    }

    @Test
    public void testGenerateOutput_emptyStr() {
        MultiLabelFactory factory = new MultiLabelFactory();
        MultiLabel output = factory.generateOutput("");
        assertEquals(0, output.getLabelSet().size());
        assertEquals("", output.getLabelString());
    }

    @Test
    public void testGenerateOutput_set() {
        MultiLabelFactory factory = new MultiLabelFactory();
        MultiLabel output = factory.generateOutput(new HashSet<>(Arrays.asList("a=true", "b=true", "c=true")));
        assertEquals(3, output.getLabelSet().size());
        assertEquals("a,b,c", output.getLabelString());

        output = factory.generateOutput(new HashSet<>(Arrays.asList("a", "b", "c")));
        assertEquals(3, output.getLabelSet().size());
        assertEquals("a,b,c", output.getLabelString());
    }

    @Test
    public void testGenerateOutput_labelSet() {
        MultiLabelFactory factory = new MultiLabelFactory();
        Set<Label> labels = new HashSet<>();
        labels.add(new Label("a"));
        labels.add(new Label("b"));
        labels.add(new Label("c"));
        MultiLabel output = factory.generateOutput(labels);
        assertEquals(3, output.getLabelSet().size());
        assertEquals("a,b,c", output.getLabelString());
    }

    @Test
    public void testGenerateOutput_emptySet() {
        MultiLabelFactory factory = new MultiLabelFactory();
        Set<String> labels = new HashSet<>();
        MultiLabel output = factory.generateOutput(labels);
        assertEquals(0, output.getLabelSet().size());
        assertEquals("", output.getLabelString());
    }

    @Test
    public void testGenerateOutput_allFalse() {
        MultiLabelFactory factory = new MultiLabelFactory();
        MultiLabel output = factory.generateOutput(new HashSet<>(Arrays.asList("a=false", "b=false", "c=false")));
        assertEquals(0, output.getLabelSet().size());
        assertEquals("", output.getLabelString());
    }

    @Test
    public void testGenerateOutput_unparseable() {
        MultiLabelFactory factory = new MultiLabelFactory();
        MultiLabel output = factory.generateOutput(new Unparseable());
        assertEquals(1, output.getLabelSet().size());
        assertTrue(output.getLabelString().startsWith("org.tribuo.multilabel.MultiLabelFactoryTest$Unparseable"));
    }

    private static final class Unparseable { }

    @Test
    public void testGenerateOutput_null() {
        MultiLabelFactory factory = new MultiLabelFactory();
        assertThrows(NullPointerException.class, () -> factory.generateOutput(null));
        assertThrows(NullPointerException.class, () -> factory.generateOutput(new HashSet<>(Arrays.asList("a", null, "b"))));
    }

}