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

package org.tribuo.classification.experiments;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableDataset;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.data.DataOptions;
import org.tribuo.data.csv.CSVLoader;
import org.tribuo.data.text.TextDataSource;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.impl.SimpleTextDataSource;
import org.tribuo.data.text.impl.TextFeatureExtractorImpl;
import org.tribuo.data.text.impl.TokenPipeline;
import org.tribuo.datasource.LibSVMDataSource;
import org.tribuo.util.Util;
import org.tribuo.util.tokens.impl.BreakIteratorTokenizer;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Test a classifier for a standard dataset.
 */
public class Test {

    private static final Logger logger = Logger.getLogger(Test.class.getName());

    /**
     * Command line options.
     */
    public static class ConfigurableTestOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Tests an already trained classifier on a dataset.";
        }

        /**
         * Hashing dimension used for standard text format.
         */
        @Option(longName = "hashing-dimension", usage = "Hashing dimension used for standard text format.")
        public int hashDim = 0;
        /**
         * Ngram size to generate when using standard text format. Defaults to 2.
         */
        @Option(longName = "ngram", usage = "Ngram size to generate when using standard text format. Defaults to 2.")
        public int ngram = 2;
        /**
         * Use term counts instead of boolean when using the standard text format.
         */
        @Option(longName = "term-counting", usage = "Use term counts instead of boolean when using the standard text format.")
        public boolean termCounting;
        /**
         * Response name in the csv file.
         */
        @Option(longName = "csv-response-name", usage = "Response name in the csv file.")
        public String csvResponseName;
        /**
         * Is the libsvm file zero indexed.
         */
        @Option(longName = "libsvm-zero-indexed", usage = "Is the libsvm file zero indexed.")
        public boolean zeroIndexed = false;
        /**
         * Load a trainer from the config file.
         */
        @Option(charName = 'f', longName = "model-path", usage = "Load a trainer from the config file.")
        public Path modelPath;
        /**
         * Path to write model predictions
         */
        @Option(charName = 'o', longName = "predictions", usage = "Path to write model predictions")
        public Path predictionPath;
        /**
         * Loads the data using the specified format. Defaults to LIBSVM.
         */
        @Option(charName = 's', longName = "input-format", usage = "Loads the data using the specified format. Defaults to LIBSVM.")
        public DataOptions.InputFormat inputFormat = DataOptions.InputFormat.LIBSVM;
        /**
         * Path to the testing file.
         */
        @Option(charName = 'v', longName = "testing-file", usage = "Path to the testing file.")
        public Path testingPath;
        /**
         * Load the model in protobuf format.
         */
        @Option(longName = "read-protobuf-model", usage = "Load the model in protobuf format.")
        public boolean protobufModel;
    }

    /**
     * Loads in the model and the dataset from the options.
     * @param o The options.
     * @return The model and the dataset.
     * @throws IOException If either the model or dataset could not be read.
     */
    @SuppressWarnings("unchecked") // deserialising generically typed datasets.
    public static Pair<Model<Label>,Dataset<Label>> load(ConfigurableTestOptions o) throws IOException {
        Path modelPath = o.modelPath;
        Path datasetPath = o.testingPath;
        logger.info(String.format("Loading model from %s", modelPath));
        Model<?> tmpModel;
        if (o.protobufModel) {
            tmpModel = Model.deserializeFromFile(modelPath);
        } else {
            try (ObjectInputStream mois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(modelPath.toFile())))) {
                tmpModel = (Model<?>) mois.readObject();
            } catch (ClassNotFoundException e) {
                throw new IllegalArgumentException("Unknown class in serialised model", e);
            }
        }
        Model<Label> model = tmpModel.castModel(Label.class);
        logger.info(String.format("Loading data from %s", datasetPath));
        Dataset<Label> test;
        switch (o.inputFormat) {
            case SERIALIZED:
                //
                // Load Tribuo serialised datasets.
                logger.info("Deserialising dataset from " + datasetPath);
                try (ObjectInputStream oits = new ObjectInputStream(new BufferedInputStream(new FileInputStream(datasetPath.toFile())))) {
                    Dataset<Label> deserTest = (Dataset<Label>) oits.readObject();
                    test = ImmutableDataset.copyDataset(deserTest,model.getFeatureIDMap(),model.getOutputIDInfo());
                    logger.info(String.format("Loaded %d testing examples for %s", test.size(), test.getOutputs().toString()));
                } catch (ClassNotFoundException e) {
                    throw new IllegalArgumentException("Unknown class in serialised dataset", e);
                }
                break;
            case SERIALIZED_PROTOBUF:
                //
                // Load Tribuo protobuf serialised datasets.
                Dataset<?> tmp = Dataset.deserializeFromFile(datasetPath);
                if (tmp.validate(Label.class)) {
                    test = Dataset.castDataset(tmp, Label.class);
                    test = ImmutableDataset.copyDataset(test,model.getFeatureIDMap(),model.getOutputIDInfo());
                    logger.info(String.format("Loaded %d testing examples for %s", test.size(), test.getOutputs().toString()));
                } else {
                    throw new IllegalArgumentException("Invalid test dataset type, expected Label.class");
                }
                break;
            case LIBSVM:
                //
                // Load the libsvm text-based data format.
                boolean zeroIndexed = o.zeroIndexed;
                int maxFeatureID = model.getFeatureIDMap().size() - 1;
                LibSVMDataSource<Label> testSVMSource = new LibSVMDataSource<>(datasetPath,new LabelFactory(),zeroIndexed,maxFeatureID);
                test = new ImmutableDataset<>(testSVMSource,model,true);
                logger.info(String.format("Loaded %d training examples for %s", test.size(), test.getOutputs().toString()));
                break;
            case TEXT:
                //
                // Using a simple Java break iterator to generate ngram features.
                TextFeatureExtractor<Label> extractor;
                if (o.hashDim > 0) {
                    extractor = new TextFeatureExtractorImpl<>(new TokenPipeline(new BreakIteratorTokenizer(Locale.US), o.ngram, o.termCounting, o.hashDim));
                } else {
                    extractor = new TextFeatureExtractorImpl<>(new TokenPipeline(new BreakIteratorTokenizer(Locale.US), o.ngram, o.termCounting));
                }

                TextDataSource<Label> testSource = new SimpleTextDataSource<>(datasetPath, new LabelFactory(), extractor);
                test = new ImmutableDataset<>(testSource, model.getFeatureIDMap(), model.getOutputIDInfo(),true);
                logger.info(String.format("Loaded %d testing examples for %s", test.size(), test.getOutputs().toString()));
                break;
            case CSV:
                //
                // Load the data using the simple CSV loader
                if (o.csvResponseName == null) {
                    throw new IllegalArgumentException("Please supply a response column name");
                }
                CSVLoader<Label> loader = new CSVLoader<>(new LabelFactory());
                test = new ImmutableDataset<>(loader.loadDataSource(datasetPath,o.csvResponseName),model.getFeatureIDMap(),model.getOutputIDInfo(),true);
                logger.info(String.format("Loaded %d testing examples for %s", test.size(), test.getOutputs().toString()));
                break;
            default:
                throw new IllegalArgumentException("Unsupported input format " + o.inputFormat);
        }
        return new Pair<>(model,test);
    }

    /**
     * Runs the Test CLI.
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        ConfigurableTestOptions o = new ConfigurableTestOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.modelPath == null || o.testingPath == null) {
            logger.info(cm.usage());
            System.exit(1);
        }
        Pair<Model<Label>,Dataset<Label>> loaded = null;
        try {
             loaded = load(o);
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Failed to load model/data", e);
            System.exit(1);
        }
        Model<Label> model = loaded.getA();
        Dataset<Label> test = loaded.getB();

        logger.info("Model is " + model.toString());
        logger.info("Labels are " + model.getOutputIDInfo().toReadableString());

        LabelEvaluator labelEvaluator = new LabelEvaluator();
        final long testStart = System.currentTimeMillis();
        List<Prediction<Label>> predictions = model.predict(test);
        LabelEvaluation evaluation = labelEvaluator.evaluate(model,predictions,test.getProvenance());
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart,testStop));
        System.out.println(evaluation.toString());
        System.out.println(evaluation.getConfusionMatrix().toString());
        if (model.generatesProbabilities()) {
            System.out.println("Average AUC = " + evaluation.averageAUCROC(false));
            System.out.println("Average weighted AUC = " + evaluation.averageAUCROC(true));
        }

        if (o.predictionPath!=null) {
            try(BufferedWriter wrt = Files.newBufferedWriter(o.predictionPath)) {
                List<String> labels = model.getOutputIDInfo().getDomain().stream().map(Label::getLabel).sorted().collect(Collectors.toList());
                wrt.write("Label,");
                wrt.write(String.join(",", labels));
                wrt.newLine();
                for(Prediction<Label> pred : predictions) {
                    Example<Label> ex = pred.getExample();
                    wrt.write(ex.getOutput().getLabel()+",");
                    wrt.write(labels
                            .stream()
                            .map(l -> Double.toString(pred
                                    .getOutputScores()
                                    .get(l).getScore()))
                            .collect(Collectors.joining(",")));
                    wrt.newLine();
                }
                wrt.flush();
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Error writing predictions", e);
            }
        }

    }
    
}
