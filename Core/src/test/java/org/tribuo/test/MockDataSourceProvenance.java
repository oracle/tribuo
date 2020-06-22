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

import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

/**
 *
 */
public class MockDataSourceProvenance implements DataSourceProvenance {
    private static final long serialVersionUID = 1L;

    public MockDataSourceProvenance() {}

    public MockDataSourceProvenance(Map<String,Provenance> map) {}

    @Override
    public String getClassName() {
        return MockDataSourceProvenance.class.getName();
    }

    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        ArrayList<Pair<String,Provenance>> list = new ArrayList<>();
        list.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,getClassName())));
        list.add(new Pair<>("source",new StringProvenance("source","test-data")));
        return list.iterator();
    }

    @Override
    public int hashCode() {
        return 42;
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof MockDataSourceProvenance;
    }

    @Override
    public String toString() {
        return generateString("DataSource");
    }
}
