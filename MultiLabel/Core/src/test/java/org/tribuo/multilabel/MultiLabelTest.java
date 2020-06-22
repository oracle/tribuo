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

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MultiLabelTest {

    @Test
    public void getsCorrectSerializableForm() {
        MultiLabel abc = new MultiLabel(mkLabelSet("a", "b", "c"));
        assertEquals("a=true,b=true,c=true", abc.getSerializableForm(false));
        assertEquals("a=true,b=true,c=true:NaN", abc.getSerializableForm(true));

        MultiLabel scored = new MultiLabel(mkLabelSet("a", "b", "c"), 1.0);
        assertEquals("a=true,b=true,c=true", scored.getSerializableForm(false));
        assertEquals("a=true,b=true,c=true:1.0", scored.getSerializableForm(true));
    }

    @Test
    public void testEqualsAndHashCode() {
        MultiLabel a = new MultiLabelFactory().generateOutput("a");
        MultiLabel b = new MultiLabelFactory().generateOutput("a");
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }

    Set<Label> mkLabelSet(String... labels) {
        Set<Label> set = new HashSet<>();
        for (String s : labels) {
            set.add(new Label(s));
        }
        return set;
    }


}