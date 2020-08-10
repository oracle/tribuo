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

import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;

/**
 * A {@link ConfigurableDataSource} base class which takes columnar data (e.g. csv or DB table rows) and generates {@link Example}s.
 */
public abstract class ColumnarDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {

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

    public Iterator<Example<T>> iterator() {
        return new InnerIterator<>(rowProcessor,rowIterator(),outputRequired);
    }

    protected abstract ColumnarIterator rowIterator();

    private static class InnerIterator<T extends Output<T>> implements Iterator<Example<T>> {
        private final boolean outputRequired;
        private final ColumnarIterator iterator;
        private final RowProcessor<T> processor;

        private Example<T> buffer = null;

        InnerIterator(RowProcessor<T> processor, ColumnarIterator iterator, boolean outputRequired) {
            this.processor = processor;
            this.iterator = iterator;
            this.outputRequired = outputRequired;
        }

        @Override
        public boolean hasNext() {
            if (buffer != null) {
                return true;
            }

            while (buffer == null && iterator.hasNext()) {
                ColumnarIterator.Row m = iterator.next();

                Optional<Example<T>> exampleOpt = processor.generateExample(m,outputRequired);
                if (exampleOpt.isPresent()) {
                    buffer = exampleOpt.get();
                }
            }
            return buffer != null;
        }

        @Override
        public Example<T> next() {
            if (hasNext()) {
                Example<T> ret = buffer;
                buffer = null;
                return ret;
            } else {
                throw new NoSuchElementException("No more data");
            }
        }
    }
}
