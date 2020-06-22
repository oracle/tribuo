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

package org.tribuo.test;

import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.SimpleDataSourceProvenance;

import java.util.ArrayList;
import java.util.List;

public class MockDataSource extends ListDataSource<MockOutput> {

    public MockDataSource(int n) {
        this(n, new MockOutputFactory());
    }

    private MockDataSource(int n, MockOutputFactory factory) {
        super(generateExamples(n),factory,new SimpleDataSourceProvenance("TestSource",factory));
    }

    private static List<Example<MockOutput>> generateExamples(int n) {
        List<Example<MockOutput>> raw = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            ArrayExample<MockOutput> e = new ArrayExample<>(new MockOutput("A"));
            e.add(new Feature("foo",1));
            raw.add(e);
        }
        return raw;
    }
}