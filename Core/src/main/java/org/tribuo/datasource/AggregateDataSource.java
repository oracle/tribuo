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
import com.oracle.labs.mlrg.olcut.provenance.primitives.EnumProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Optional;

/**
 * Aggregates multiple {@link DataSource}s, uses {@link AggregateDataSource.IterationOrder} to control the
 * iteration order.
 */
public class AggregateDataSource<T extends Output<T>> implements DataSource<T> {

    /**
     * Specifies the iteration order of the inner sources.
     */
    public enum IterationOrder {
        /**
         * Round-robins the iterators (i.e., chooses one from each in turn).
         */
        ROUNDROBIN,
        /**
         * Iterates each dataset sequentially, in the order of the sources list.
         */
        SEQUENTIAL;
    }

    private final IterationOrder order;

    private final List<DataSource<T>> sources;

    /**
     * Creates an aggregate data source which will iterate the provided
     * sources in the order of the list (i.e., using {@link IterationOrder#SEQUENTIAL}.
     * @param sources The sources to aggregate.
     */
    public AggregateDataSource(List<DataSource<T>> sources) {
        this(sources,IterationOrder.SEQUENTIAL);
    }

    /**
     * Creates an aggregate data source using the supplied sources and iteration order.
     * @param sources The sources to iterate.
     * @param order The iteration order.
     */
    public AggregateDataSource(List<DataSource<T>> sources, IterationOrder order) {
        this.sources = Collections.unmodifiableList(new ArrayList<>(sources));
        this.order = order;
    }
    
    @Override
    public String toString() {
        return "AggregateDataSource(sources="+sources.toString()+",order="+order+")";
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return sources.get(0).getOutputFactory();
    }

    @Override
    public Iterator<Example<T>> iterator() {
        switch (order) {
            case ROUNDROBIN:
                return new ADSRRIterator<>(sources);
            case SEQUENTIAL:
                return new ADSSeqIterator<>(sources);
            default:
                throw new IllegalStateException("Unknown enum value " + order);
        }
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new AggregateDataSourceProvenance(this);
    }

    static class ADSRRIterator<T extends Output<T>> implements Iterator<Example<T>> {
        private final Deque<Iterator<Example<T>>> queue;

        ADSRRIterator(List<? extends DataSource<T>> sources) {
            this.queue = new ArrayDeque<>(sources.size());
            for (DataSource<T> ds : sources) {
                Iterator<Example<T>> itr = ds.iterator();
                if (itr.hasNext()) {
                    queue.addLast(itr);
                }
            }
        }

        @Override
        public boolean hasNext() {
            return !queue.isEmpty();
        }

        @Override
        public Example<T> next() {
            if (!hasNext()) {
                throw new NoSuchElementException("Iterator exhausted");
            }
            Iterator<Example<T>> itr = queue.pollFirst();
            if (itr.hasNext()) {
                Example<T> buff = itr.next();
                if (itr.hasNext()) {
                    queue.addLast(itr);
                }
                return buff;
            } else {
                throw new IllegalStateException("Invalid iterator in queue");
            }
        }
    }

    static class ADSSeqIterator<T extends Output<T>> implements Iterator<Example<T>> {
        private final Iterator<? extends DataSource<T>> si;
        private Iterator<Example<T>> curr;

        ADSSeqIterator(List<? extends DataSource<T>> sources) {
            this.si = sources.iterator();
            this.curr = null;
        }

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

    /**
     * Provenance for the {@link AggregateDataSource}.
     */
    public static class AggregateDataSourceProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private static final String SOURCES = "sources";
        private static final String ORDER = "order";

        private final StringProvenance className;
        private final ListProvenance<DataSourceProvenance> provenances;
        private EnumProvenance<IterationOrder> orderProvenance;

        /**
         * Constructs an AggregateDataSourceProvenance from the host object.
         * @param host The host data source.
         * @param <T> The data source type.
         */
        <T extends Output<T>> AggregateDataSourceProvenance(AggregateDataSource<T> host) {
            this.className = new StringProvenance(CLASS_NAME,host.getClass().getName());
            this.provenances = ListProvenance.createListProvenance(host.sources);
            this.orderProvenance = new EnumProvenance<>(ORDER,host.order);
        }

        /**
         * Deserialization constructor.
         * @param map The provenance map.
         */
        @SuppressWarnings({"unchecked","rawtypes"}) //ListProvenance cast, EnumProvenance cast
        public AggregateDataSourceProvenance(Map<String,Provenance> map) {
            this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME, StringProvenance.class,AggregateDataSourceProvenance.class.getSimpleName());
            this.provenances = ObjectProvenance.checkAndExtractProvenance(map,SOURCES,ListProvenance.class,AggregateDataSourceProvenance.class.getSimpleName());

            // Provenance added in Tribuo 4.1
            Optional<EnumProvenance> opt = ObjectProvenance.maybeExtractProvenance(map,ORDER,EnumProvenance.class,AggregateDataSourceProvenance.class.getSimpleName());
            this.orderProvenance = opt.orElseGet(() -> new EnumProvenance<>(ORDER, IterationOrder.SEQUENTIAL));
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
            list.add(new Pair<>(ORDER,getOrder()));

            return list.iterator();
        }

        private EnumProvenance<IterationOrder> getOrder() {
            if (orderProvenance != null) {
                return orderProvenance;
            } else {
                return new EnumProvenance<>(ORDER,IterationOrder.SEQUENTIAL);
            }
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof AggregateDataSourceProvenance)) return false;
            AggregateDataSourceProvenance pairs = (AggregateDataSourceProvenance) o;
            return className.equals(pairs.className) &&
                    provenances.equals(pairs.provenances) &&
                    getOrder().equals(pairs.getOrder());
        }

        @Override
        public int hashCode() {
            return Objects.hash(className, provenances, getOrder());
        }

        @Override
        public String toString() {
            return generateString("DataSource");
        }
    }
}
