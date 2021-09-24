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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.datasource.AggregateDataSource.IterationOrder;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Aggregates multiple {@link ConfigurableDataSource}s, uses {@link AggregateDataSource.IterationOrder} to control the
 * iteration order.
 * <p>
 * Identical to {@link AggregateDataSource} except it can be configured.
 */
public class AggregateConfigurableDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {

    @Config(mandatory = true, description = "The iteration order.")
    private IterationOrder order;

    @Config(mandatory = true, description = "The sources to aggregate.")
    private List<ConfigurableDataSource<T>> sources;

    /**
     * Creates an aggregate data source which will iterate the provided
     * sources in the order of the list (i.e., using {@link IterationOrder#SEQUENTIAL}.
     * @param sources The sources to aggregate.
     */
    public AggregateConfigurableDataSource(List<ConfigurableDataSource<T>> sources) {
        this(sources, IterationOrder.SEQUENTIAL);
    }

    /**
     * Creates an aggregate data source using the supplied sources and iteration order.
     * @param sources The sources to iterate.
     * @param order The iteration order.
     */
    public AggregateConfigurableDataSource(List<ConfigurableDataSource<T>> sources, IterationOrder order) {
        this.sources = Collections.unmodifiableList(new ArrayList<>(sources));
        this.order = order;
    }
    
    @Override
    public String toString() {
        return "AggregateConfigurableDataSource(sources="+sources.toString()+",order="+order+")";
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return sources.get(0).getOutputFactory();
    }

    @Override
    public Iterator<Example<T>> iterator() {
        switch (order) {
            case ROUNDROBIN:
                return new AggregateDataSource.ADSRRIterator<>(sources);
            case SEQUENTIAL:
                return new AggregateDataSource.ADSSeqIterator<>(sources);
            default:
                throw new IllegalStateException("Unknown enum value " + order);
        }
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return new AggregateConfigurableDataSourceProvenance(this);
    }

    /**
     * Provenance for the {@link AggregateConfigurableDataSource}.
     */
    public static class AggregateConfigurableDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * Creates a AggregateConfigurableDataSourceProvenance from the host object.
         * @param host The data source to provenance.
         * @param <T> The type of the data source.
         */
        <T extends Output<T>> AggregateConfigurableDataSourceProvenance(AggregateConfigurableDataSource<T> host) {
            super(host, "DataSource");
        }

        /**
         * Deserialization constructor.
         * @param map The provenance to deserialize.
         */
        public AggregateConfigurableDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        /**
         * Inner deserialization constructor.
         * @param info The extracted information.
         */
        private AggregateConfigurableDataSourceProvenance(ExtractedInfo info) {
            super(info);
        }

        /**
         * Extracts the class name and host type fields from the provenance map.
         * @param map The provenance map.
         * @return The extracted information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String, Provenance> map) {
            Map<String, Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, AggregateConfigurableDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters, HOST_SHORT_NAME, StringProvenance.class, AggregateConfigurableDataSourceProvenance.class.getSimpleName()).getValue();
            return new ExtractedInfo(className, hostTypeStringName, configuredParameters, Collections.emptyMap());
        }
    }
}
