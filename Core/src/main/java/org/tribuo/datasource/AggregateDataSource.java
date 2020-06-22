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

package org.tribuo.datasource;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * Aggregates multiple {@link DataSource}s, and round-robins the iterators.
 */
public class AggregateDataSource<T extends Output<T>> implements DataSource<T> {
    
    private final List<DataSource<T>> sources;

    public AggregateDataSource(List<DataSource<T>> sources) {
        this.sources = Collections.unmodifiableList(new ArrayList<>(sources));
    }
    
    @Override
    public String toString() {
        return "AggregateDataSource(sources="+sources.toString()+")";
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return sources.get(0).getOutputFactory();
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return new ADSIterator();
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new AggregateDataSourceProvenance(this);
    }

    private class ADSIterator implements Iterator<Example<T>> {
        Iterator<DataSource<T>> si = sources.iterator();
        Iterator<Example<T>> curr = null;
        @Override
        public boolean hasNext() {
            if (curr == null) {
                if(si.hasNext()) {
                    DataSource<T> nds = si.next();
                    curr = nds.iterator();
                    return hasNext();
                } else {
                    return false;
                }
            } else {
                if(curr.hasNext()) {
                    return true;
                } else {
                    curr = null;
                    return hasNext();
                }
            }
        }

        @Override
        public Example<T> next() {
            if (hasNext()) {
                return curr.next();
            } else {
                throw new NoSuchElementException("No more data");
            }
        }
    }

    public static class AggregateDataSourceProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private static final String SOURCES = "sources";

        private final StringProvenance className;
        private final ListProvenance<DataSourceProvenance> provenances;

        <T extends Output<T>> AggregateDataSourceProvenance(AggregateDataSource<T> host) {
            this.className = new StringProvenance(CLASS_NAME,host.getClass().getName());
            this.provenances = ListProvenance.createListProvenance(host.sources);
        }

        @SuppressWarnings("unchecked") //ListProvenance cast
        public AggregateDataSourceProvenance(Map<String,Provenance> map) {
            this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME, StringProvenance.class,AggregateDataSourceProvenance.class.getSimpleName());
            this.provenances = ObjectProvenance.checkAndExtractProvenance(map,SOURCES,ListProvenance.class,AggregateDataSourceProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return className.getValue();
        }

        @Override
        public Iterator<Pair<String, Provenance>> iterator() {
            ArrayList<Pair<String,Provenance>> list = new ArrayList<>();

            list.add(new Pair<>(CLASS_NAME,className));
            list.add(new Pair<>(SOURCES,provenances));

            return list.iterator();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof AggregateDataSourceProvenance)) return false;
            AggregateDataSourceProvenance pairs = (AggregateDataSourceProvenance) o;
            return className.equals(pairs.className) &&
                    provenances.equals(pairs.provenances);
        }

        @Override
        public int hashCode() {
            return Objects.hash(className, provenances);
        }

        @Override
        public String toString() {
            return generateString("DataSource");
        }
    }
}
