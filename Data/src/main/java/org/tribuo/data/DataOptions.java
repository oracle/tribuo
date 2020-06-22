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

package org.tribuo.data;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.ImmutableDataset;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Trainer;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.csv.CSVDataSource;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.text.TextDataSource;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.impl.SimpleTextDataSource;
import org.tribuo.data.text.impl.TextFeatureExtractorImpl;
import org.tribuo.data.text.impl.TokenPipeline;
import org.tribuo.dataset.MinimumCardinalityDataset;
import org.tribuo.datasource.LibSVMDataSource;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.Locale;
import java.util.logging.Logger;

/**
 * Options for working with training and test data in a CLI.
 */
public final class DataOptions implements Options {
    private static final Logger logger = Logger.getLogger(DataOptions.class.getName());

    public enum InputFormat {
        SERIALIZED, LIBSVM, TEXT, CSV, COLUMNAR
    }

    public enum Delimiter {
        COMMA(','), TAB('\t'), SEMICOLON(';');

        public final char value;
        Delimiter(char value) {
            this.value = value;
        }
    }

    @Override
    public String getOptionsDescription() {
        return "Options for loading and processing train and test data.";
    }

    @Option(longName="hashing-dimension",usage="Hashing dimension used for standard text format.")
    public int hashDim = 0;
    @Option(longName="ngram",usage="Ngram size to generate when using standard text format.")
    public int ngram = 2;
    @Option(longName="term-counting",usage="Use term counts instead of boolean when using the standard text format.")
    public boolean termCounting;
    @Option(charName='f',longName="model-output-path",usage="Path to serialize model to.")
    public Path outputPath;
    @Option(charName='r',longName="seed",usage="RNG seed.")
    public long seed = Trainer.DEFAULT_SEED;
    @Option(charName='s',longName="input-format",usage="Loads the data using the specified format.")
    public InputFormat inputFormat = InputFormat.LIBSVM;
    @Option(longName="csv-response-name",usage="Response name in the csv file.")
    public String csvResponseName;
    @Option(longName="csv-delimiter",usage="Delimiter")
    public Delimiter delimiter = Delimiter.COMMA;
    @Option(longName="csv-quote-char",usage="Quote character in the CSV file.")
    public char csvQuoteChar = '"';
    @Option(longName="columnar-row-processor",usage="The name of the row processor from the config file.")
    public RowProcessor<?> rowProcessor;
    @Option(longName="min-count",usage="Minimum cardinality of the features.")
    public int minCount = 0;
    @Option(charName='u',longName="training-file",usage="Path to the training file.")
    public Path trainingPath;
    @Option(charName='v',longName="testing-file",usage="Path to the testing file.")
    public Path testingPath;

    public <T extends Output<T>> Pair<Dataset<T>,Dataset<T>> load(OutputFactory<T> outputFactory) throws IOException {
        logger.info(String.format("Loading data from %s", trainingPath));
        Dataset<T> train;
        Dataset<T> test;
        char separator;
        switch (inputFormat) {
            case SERIALIZED:
                //
                // Load Tribuo serialised datasets.
                logger.info("Deserialising dataset from " + trainingPath);
                try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(trainingPath.toFile())));
                     ObjectInputStream oits = new ObjectInputStream(new BufferedInputStream(new FileInputStream(testingPath.toFile())))) {
                    @SuppressWarnings("unchecked")
                    Dataset<T> tmp = (Dataset<T>) ois.readObject();
                    train = tmp;
                    if (minCount > 0) {
                        logger.info("Found " + train.getFeatureIDMap().size() + " features");
                        logger.info("Removing features that occur fewer than " + minCount + " times.");
                        train = new MinimumCardinalityDataset<>(train,minCount);
                    }
                    logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                    logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                    @SuppressWarnings("unchecked")
                    Dataset<T> deserTest = (Dataset<T>) oits.readObject();
                    test = new ImmutableDataset<>(deserTest,deserTest.getSourceProvenance(),deserTest.getOutputFactory(),train.getFeatureIDMap(),train.getOutputIDInfo(),true);
                } catch (ClassNotFoundException e) {
                    throw new IllegalArgumentException("Unknown class in serialised files", e);
                }
                break;
            case LIBSVM:
                //
                // Load the libsvm text-based data format.
                LibSVMDataSource<T> trainSVMSource = new LibSVMDataSource<>(trainingPath,outputFactory);
                train = new MutableDataset<>(trainSVMSource);
                boolean zeroIndexed = trainSVMSource.isZeroIndexed();
                int maxFeatureID = trainSVMSource.getMaxFeatureID();
                if (minCount > 0) {
                    logger.info("Removing features that occur fewer than " + minCount + " times.");
                    train = new MinimumCardinalityDataset<>(train,minCount);
                }
                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                test = new ImmutableDataset<>(new LibSVMDataSource<>(testingPath,outputFactory,zeroIndexed,maxFeatureID), train.getFeatureIDMap(), train.getOutputIDInfo(), false);
                break;
            case TEXT:
                //
                // Using a simple Java break iterator to generate ngram features.
                TextFeatureExtractor<T> extractor;
                if (hashDim > 0) {
                    extractor = new TextFeatureExtractorImpl<>(new TokenPipeline(new BreakIteratorTokenizer(Locale.US), ngram, termCounting, hashDim));
                } else {
                    extractor = new TextFeatureExtractorImpl<>(new TokenPipeline(new BreakIteratorTokenizer(Locale.US), ngram, termCounting));
                }

                TextDataSource<T> trainSource = new SimpleTextDataSource<>(trainingPath, outputFactory, extractor);
                train = new MutableDataset<>(trainSource);
                if (minCount > 0) {
                    logger.info("Removing features that occur fewer than " + minCount + " times.");
                    train = new MinimumCardinalityDataset<>(train,minCount);
                }

                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");

                TextDataSource<T> testSource = new SimpleTextDataSource<>(testingPath, outputFactory, extractor);
                test = new ImmutableDataset<>(testSource, train.getFeatureIDMap(), train.getOutputIDInfo(), false);
                break;
            case CSV:
                //
                // Load the data using the simple CSV loader
                if (csvResponseName == null) {
                    throw new IllegalArgumentException("Please supply a response column name");
                }
                separator = delimiter.value;
                CSVLoader<T> loader = new CSVLoader<>(separator,outputFactory);
                train = new MutableDataset<>(loader.loadDataSource(trainingPath,csvResponseName));
                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                test = new MutableDataset<>(loader.loadDataSource(testingPath,csvResponseName));
                break;
            case COLUMNAR:
                if (rowProcessor == null) {
                    throw new IllegalArgumentException("Please supply a RowProcessor");
                }
                OutputFactory<?> rowOutputFactory = rowProcessor.getResponseProcessor().getOutputFactory();
                if (!rowOutputFactory.equals(outputFactory)) {
                    throw new IllegalArgumentException("The RowProcessor doesn't use the same kind of OutputFactory as the one supplied. RowProcessor has " + rowOutputFactory.getClass().getSimpleName() + ", supplied " + outputFactory.getClass().getName());
                }
                @SuppressWarnings("unchecked") // checked by the if statement above
                RowProcessor<T> typedRowProcessor = (RowProcessor<T>) rowProcessor;
                separator = delimiter.value;
                train = new MutableDataset<>(new CSVDataSource<>(trainingPath,typedRowProcessor,true,separator,csvQuoteChar));
                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                test = new MutableDataset<>(new CSVDataSource<>(testingPath,typedRowProcessor,true,separator,csvQuoteChar));
                break;
            default:
                throw new IllegalArgumentException("Unsupported input format " + inputFormat);
        }
        logger.info(String.format("Loaded %d testing examples", test.size()));
        return new Pair<>(train,test);
    }

    public <T extends Output<T>> void saveModel(Model<T> model) throws IOException {
        FileOutputStream fout = new FileOutputStream(outputPath.toFile());
        ObjectOutputStream oout = new ObjectOutputStream(fout);
        oout.writeObject(model);
        oout.close();
        fout.close();
        logger.info("Serialized model to file: " + outputPath);
    }
}
