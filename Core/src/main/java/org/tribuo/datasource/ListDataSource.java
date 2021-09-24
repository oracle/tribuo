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

import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * A data source which wraps up a list of {@link Example}s
 * along with their {@link DataSourceProvenance} and an {@link OutputFactory}.
 * <p>
 * Used for machine generated data, or some other place where it's difficult to
 * write a specific data source.
 */
public class ListDataSource<T extends Output<T>> implements DataSource<T> {

    private final List<Example<T>> data;

    private final OutputFactory<T> factory;

    private final DataSourceProvenance provenance;

    /**
     * Constructs an in-memory data source wrapping the supplied examples.
     * @param list The examples.
     * @param factory The output factory.
     * @param provenance The data source provenance.
     */
    public ListDataSource(List<Example<T>> list, OutputFactory<T> factory, DataSourceProvenance provenance) {
        this.data = Collections.unmodifiableList(new ArrayList<>(list));
        this.factory = factory;
        this.provenance = provenance;
    }

    /**
     * Number of examples.
     * @return The number of examples.
     */
    public int size() {
        return data.size();
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return factory;
    }

    @Override
    public DataSourceProvenance getProvenance() {
        return provenance;
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return data.iterator();
    }

    @Override
    public String toString() {
        return provenance.toString();
    }
}
