/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.transform.TransformationMap;
import org.tribuo.transform.TransformerMap;
import org.tribuo.transform.transformations.LinearScalingTransformation;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Locale;
import java.util.logging.Logger;

/**
 * Options for working with training and test data in a CLI.
 */
public final class DataOptions implements Options {
    private static final Logger logger = Logger.getLogger(DataOptions.class.getName());

    /**
     * The input formats supported by this options object.
     */
    public enum InputFormat {
        /**
         * Serialized Tribuo datasets.
         */
        SERIALIZED,
        /**
         * Protobuf serialized Tribuo datasets.
         */
        SERIALIZED_PROTOBUF,
        /**
         * LibSVM/svm-light format data.
         */
        LIBSVM,
        /**
         * Text data in Tribuo's standard format (i.e., each line is "output ## text data").
         */
        TEXT,
        /**
         * Simple numeric CSV file.
         */
        CSV,
        /**
         * A CSV file parsed using a configured {@link RowProcessor}.
         */
        COLUMNAR
    }

    /**
     * The delimiters supported by CSV files in this options object.
     */
    public enum Delimiter {
        /**
         * Comma separator.
         */
        COMMA(','),
        /**
         * Tab separator.
         */
        TAB('\t'),
        /**
         * Semicolon separator.
         */
        SEMICOLON(';');

        /**
         * The delimiter character.
         */
        public final char value;

        Delimiter(char value) {
            this.value = value;
        }
    }

    @Override
    public String getOptionsDescription() {
        return "Options for loading and processing train and test data.";
    }

    /**
     * Hashing dimension used for standard text format.
     */
    @Option(longName = "hashing-dimension", usage = "Hashing dimension used for standard text format.")
    public int hashDim = 0;
    /**
     * Ngram size to generate when using standard text format.
     */
    @Option(longName = "ngram", usage = "Ngram size to generate when using standard text format.")
    public int ngram = 2;
    /**
     * Use term counts instead of boolean when using the standard text format.
     */
    @Option(longName = "term-counting", usage = "Use term counts instead of boolean when using the standard text format.")
    public boolean termCounting;
    /**
     * Path to serialize model to.
     */
    @Option(charName = 'f', longName = "model-output-path", usage = "Path to serialize model to.")
    public Path outputPath;
    /**
     * Write the model out as a protobuf.
     */
    @Option(longName = "model-output-protobuf", usage = "Serialize the model as a protobuf.")
    public boolean modelOutputProtobuf;
    /**
     * RNG seed.
     */
    @Option(charName = 'r', longName = "seed", usage = "RNG seed.")
    public long seed = Trainer.DEFAULT_SEED;
    /**
     * Loads the data using the specified format.
     */
    @Option(charName = 's', longName = "input-format", usage = "Loads the data using the specified format.")
    public InputFormat inputFormat = InputFormat.LIBSVM;
    /**
     * Response name in the csv file.
     */
    @Option(longName = "csv-response-name", usage = "Response name in the csv file.")
    public String csvResponseName;
    /**
     * Delimiter
     */
    @Option(longName = "csv-delimiter", usage = "Delimiter")
    public Delimiter delimiter = Delimiter.COMMA;
    /**
     * Quote character in the CSV file.
     */
    @Option(longName = "csv-quote-char", usage = "Quote character in the CSV file.")
    public char csvQuoteChar = '"';
    /**
     * The name of the row processor from the config file.
     */
    @Option(longName = "columnar-row-processor", usage = "The name of the row processor from the config file.")
    public RowProcessor<?> rowProcessor;
    /**
     * Minimum cardinality of the features.
     */
    @Option(longName = "min-count", usage = "Minimum cardinality of the features.")
    public int minCount = 0;
    /**
     * Path to the training file.
     */
    @Option(charName = 'u', longName = "training-file", usage = "Path to the training file.")
    public Path trainingPath;
    /**
     * Path to the testing file.
     */
    @Option(charName = 'v', longName = "testing-file", usage = "Path to the testing file.")
    public Path testingPath;
    /**
     * Scales the features to the range 0-1 independently.
     */
    @Option(longName="scale-features",usage="Scales the features to the range 0-1 independently.")
    public boolean scaleFeatures;
    /**
     * Includes implicit zeros in the scale range calculation.
     */
    @Option(longName="scale-including-zeros",usage="Includes implicit zeros in the scale range calculation.")
    public boolean scaleIncZeros;

    /**
     * Loads the training and testing data from {@link #trainingPath} and {@link #testingPath}
     * according to the other parameters specified in this class.
     * @param outputFactory The output factory to use to process the inputs.
     * @param <T> The dataset output type.
     * @return A pair containing the training and testing datasets. The training dataset is element 'A' and the
     * testing dataset is element 'B'.
     * @throws IOException If the paths could not be loaded.
     */
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
                        train = new MinimumCardinalityDataset<>(train, minCount);
                    }
                    logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                    logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                    logger.info("Deserialising dataset from " + testingPath);
                    @SuppressWarnings("unchecked")
                    Dataset<T> deserTest = (Dataset<T>) oits.readObject();
                    test = new ImmutableDataset<>(deserTest, deserTest.getSourceProvenance(), deserTest.getOutputFactory(), train.getFeatureIDMap(), train.getOutputIDInfo(), true);
                } catch (ClassNotFoundException e) {
                    throw new IllegalArgumentException("Unknown class in serialised files", e);
                }
                break;
            case SERIALIZED_PROTOBUF:
                //
                // Load Tribuo protobuf serialised datasets.
                logger.info("Deserialising protobuf dataset from " + trainingPath);
                Dataset<?> tmp = Dataset.deserializeFromFile(trainingPath);
                if (tmp.validate(outputFactory.getTypeWitness())) {
                    train = Dataset.castDataset(tmp, outputFactory.getTypeWitness());
                    if (minCount > 0) {
                        logger.info("Found " + train.getFeatureIDMap().size() + " features");
                        logger.info("Removing features that occur fewer than " + minCount + " times.");
                        train = new MinimumCardinalityDataset<>(train, minCount);
                    }
                    logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                    logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");

                    logger.info("Deserialising protobuf dataset from " + testingPath);
                    tmp = Dataset.deserializeFromFile(testingPath);
                    if (tmp.validate(outputFactory.getTypeWitness())) {
                        Dataset<T> deserTest = Dataset.castDataset(tmp, outputFactory.getTypeWitness());
                        test = new ImmutableDataset<>(deserTest, deserTest.getSourceProvenance(), deserTest.getOutputFactory(), train.getFeatureIDMap(), train.getOutputIDInfo(), true);
                    } else {
                        throw new IllegalArgumentException("Invalid test dataset type, expected " + outputFactory.getUnknownOutput().getClass());
                    }
                } else {
                    throw new IllegalArgumentException("Invalid train dataset type, expected " + outputFactory.getUnknownOutput().getClass());
                }
                break;
            case LIBSVM:
                //
                // Load the libsvm text-based data format.
                LibSVMDataSource<T> trainSVMSource = new LibSVMDataSource<>(trainingPath, outputFactory);
                train = new MutableDataset<>(trainSVMSource);
                boolean zeroIndexed = trainSVMSource.isZeroIndexed();
                int maxFeatureID = trainSVMSource.getMaxFeatureID();
                if (minCount > 0) {
                    logger.info("Removing features that occur fewer than " + minCount + " times.");
                    train = new MinimumCardinalityDataset<>(train, minCount);
                }
                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                test = new ImmutableDataset<>(new LibSVMDataSource<>(testingPath, outputFactory, zeroIndexed, maxFeatureID), train.getFeatureIDMap(), train.getOutputIDInfo(), false);
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
                    train = new MinimumCardinalityDataset<>(train, minCount);
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
                CSVLoader<T> loader = new CSVLoader<>(separator, outputFactory);
                train = new MutableDataset<>(loader.loadDataSource(trainingPath, csvResponseName));
                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                test = new MutableDataset<>(loader.loadDataSource(testingPath, csvResponseName));
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
                train = new MutableDataset<>(new CSVDataSource<>(trainingPath, typedRowProcessor, true, separator, csvQuoteChar));
                logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
                logger.info("Found " + train.getFeatureIDMap().size() + " features, and " + train.getOutputInfo().size() + " response dimensions");
                test = new MutableDataset<>(new CSVDataSource<>(testingPath, typedRowProcessor, true, separator, csvQuoteChar));
                break;
            default:
                throw new IllegalArgumentException("Unsupported input format " + inputFormat);
        }
        logger.info(String.format("Loaded %d testing examples", test.size()));
        if (scaleFeatures) {
            logger.info("Fitting feature scaling");
            TransformationMap map = new TransformationMap(Collections.singletonList(new LinearScalingTransformation()));
            TransformerMap transformers = train.createTransformers(map,scaleIncZeros);
            logger.info("Applying scaling to training dataset");
            train = transformers.transformDataset(train);
            logger.info("Applying scaling to testing dataset");
            test = transformers.transformDataset(test);
        }
        return new Pair<>(train,test);
    }

    /**
     * Saves the model out to the path in {@link #outputPath}.
     * @param model The model to save.
     * @param <T> The model's output type.
     * @throws IOException If the model could not be saved.
     */
    public <T extends Output<T>> void saveModel(Model<T> model) throws IOException {
        if (modelOutputProtobuf) {
            model.serializeToFile(outputPath);
            logger.info("Serialized model as a protobuf to file: " + outputPath);
        } else {
            try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(outputPath))) {
                oos.writeObject(model);
                logger.info("Serialized model to file: " + outputPath);
            }
        }
    }
}

