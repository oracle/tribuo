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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;

import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * A base class for textual data sets. We assume that all textual data is
 * written and read using UTF-8.
 */
public abstract class TextDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {

    /**
     * Document preprocessors that should be run on the documents that make up
     * this data set.
     */
    @Config(description = "The document preprocessors to run on each document in the data source.")
    protected List<DocumentPreprocessor> preprocessors = new ArrayList<>();

    /**
     * The path that data was read from.
     */
    @Config(mandatory = true, description = "The path to read the data from.")
    protected Path path;

    /**
     * The factory that converts a String into an {@link Output}.
     */
    @Config(mandatory = true, description = "The factory that converts a String into an Output instance.")
    protected OutputFactory<T> outputFactory;

    /**
     * The extractor that we'll use to turn text into examples.
     */
    @Config(mandatory = true, description = "The feature extractor that generates Features from text.")
    protected TextFeatureExtractor<T> extractor;

    /**
     * The actual data read out of the text file.
     */
    protected final List<Example<T>> data = new ArrayList<>();

    /**
     * for olcut
     */
    protected TextDataSource() {
    }

    /**
     * Creates a text data set by reading it from a path.
     *
     * @param path          The path to read data from
     * @param outputFactory The output factory used to generate the outputs.
     * @param extractor     The feature extractor to run on the text.
     * @param preprocessors Processors that will be run on the data before it
     *                      is added as examples.
     */
    public TextDataSource(Path path, OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor, DocumentPreprocessor... preprocessors) {
        this.path = path;
        this.outputFactory = outputFactory;
        this.extractor = extractor;
        this.preprocessors.addAll(Arrays.asList(preprocessors));
    }

    /**
     * Creates a text data set by reading it from a file.
     *
     * @param file          The file to read data from
     * @param outputFactory The output factory used to generate the outputs.
     * @param extractor     The feature extractor to run on the text.
     * @param preprocessors Processors that will be run on the data before it
     *                      is added as examples.
     */
    public TextDataSource(File file, OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor, DocumentPreprocessor... preprocessors) {
        this(file.toPath(), outputFactory, extractor, preprocessors);
    }

    @Override
    public Iterator<Example<T>> iterator() {
        if (!data.isEmpty()) {
            return data.iterator();
        } else {
            throw new IllegalStateException("read was not called in " + this.getClass().getName());
        }
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();

        buffer.append(this.getClass().getSimpleName());
        buffer.append("(path=");
        buffer.append(path.toString());
        buffer.append(",extractor=");
        buffer.append(extractor.toString());
        buffer.append(",preprocessors=");
        buffer.append(preprocessors.toString());
        buffer.append(")");

        return buffer.toString();
    }

    /**
     * A method that can be overridden to do different things to each document
     * that we've read. By default iterates the preprocessors and applies them to the document.
     *
     * @param doc The document to handle
     * @return a (possibly modified) version of the document.
     */
    protected String handleDoc(String doc) {
        String newDoc = doc;
        for (DocumentPreprocessor p : preprocessors) {
            newDoc = p.processDoc(newDoc);
        }
        return newDoc;
    }

    /**
     * Reads the data from the Path.
     *
     * @throws java.io.IOException if there is any error reading the data.
     */
    protected abstract void read() throws java.io.IOException;

    /**
     * Returns the output factory used to convert the text input into an {@link Output}.
     *
     * @return The output factory.
     */
    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

}
