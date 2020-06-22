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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A {@link ConfigurableDataSource} base class which takes columnar data (e.g. csv or DB table rows) and generates {@link Example}s.
 */
public abstract class ColumnarDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {

    private static final Logger logger = Logger.getLogger(ColumnarDataSource.class.getName());

    @Config(mandatory = true,description="The output factory to use.")
    private OutputFactory<T> outputFactory;

    @Config(mandatory = true,description="The row processor to use.")
    protected RowProcessor<T> rowProcessor;

    @Config(description="Is an output required from each row?")
    protected boolean outputRequired = true;

    /**
     * For OLCUT.
     */
    protected ColumnarDataSource() {}

    public ColumnarDataSource(OutputFactory<T> outputFactory, RowProcessor<T> rowProcessor, boolean outputRequired) {
        this.outputFactory = outputFactory;
        this.rowProcessor = rowProcessor;
        this.outputRequired = outputRequired;
    }

    /**
     * Returns the metadata keys and value types that are created
     * by this DataSource.
     * @return The metadata keys and value types.
     */
    public Map<String,Class<?>> getMetadataTypes() {
        return rowProcessor.getMetadataTypes();
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    public abstract ConfiguredDataSourceProvenance getProvenance();

    protected class ColumnarIterator implements Iterator<Example<T>> {

        final Iterator<Map<String,String>> itr;
        final String[] headers;
        Example<T> example;

        int nr = 0;

        public ColumnarIterator(Iterator<Map<String,String>> itr, String[] headers) {
            this.itr = itr;
            this.headers = headers;
            Set<String> headerSet = new LinkedHashSet<>(Arrays.asList(headers));

            // If the row processor has not been configured with the headers, configure it.
            if (!rowProcessor.isConfigured()) {
                rowProcessor.expandRegexMapping(headerSet);
            }

            Set<String> columns = rowProcessor.getColumnNames();
            if(!headerSet.containsAll(columns)) {
                Set<String> missingProcessor = new HashSet<>(columns);
                missingProcessor.removeAll(headerSet);
                throw new IllegalArgumentException("Processor fields have no matching fields in data: " + String.join(", ", missingProcessor));
            }
        }

        public <Iter extends Iterator<Map<String, String>> & FieldNames> ColumnarIterator(Iter itr) {
            this(itr, itr.fields());
        }

        @Override
        public boolean hasNext() {
            if (example != null) {
                return true;
            }

            while (example == null && itr.hasNext()) {
                Map<String, String> m = Collections.unmodifiableMap(itr.next());
                nr++;
                if (nr % 50_000 == 0) {
                    logger.info(String.format("Read %,d", nr));
                }

                Optional<Example<T>> exampleOpt = rowProcessor.generateExample(nr,m,outputRequired);
                if (exampleOpt.isPresent()) {
                    example = exampleOpt.get();
                }
            }
            return example != null;
        }

        @Override
        public Example<T> next() {
            if (hasNext()) {
                Example<T> ret = example;
                example = null;
                return ret;
            } else {
                throw new NoSuchElementException("No more data");
            }
        }
    }

}
