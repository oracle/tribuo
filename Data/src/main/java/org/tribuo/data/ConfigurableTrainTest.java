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

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Trainer;
import org.tribuo.evaluation.CrossValidation;
import org.tribuo.evaluation.DescriptiveStats;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.evaluation.EvaluationAggregator;
import org.tribuo.evaluation.Evaluator;
import org.tribuo.evaluation.metrics.MetricID;
import org.tribuo.transform.TransformTrainer;
import org.tribuo.transform.TransformationMap;
import org.tribuo.util.Util;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Build and run a predictor for a standard dataset.
 */
public final class ConfigurableTrainTest {

    private static final Logger logger = Logger.getLogger(ConfigurableTrainTest.class.getName());

    private ConfigurableTrainTest() {}

    /**
     * Command line options.
     */
    public static class ConfigurableTrainTestOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Loads a Trainer from a config file, trains a Model (optionally with cross-validation), tests it and optionally saves it to disk.";
        }

        /**
         * Data loading options.
         */
        public DataOptions general;

        /**
         * Load a trainer from the config file.
         */
        @Option(charName = 't', longName = "trainer", usage = "Load a trainer from the config file.")
        public Trainer<?> trainer;

        /**
         * Load a transformation map from the config file.
         */
        @Option(longName = "transformer", usage = "Load a transformation map from the config file.")
        public TransformationMap transformationMap;

        /**
         * The output factory to construct.
         */
        @Option(charName = 'a', longName = "output-factory", usage = "The output factory to construct.")
        public OutputFactory<?> outputFactory;

        /**
         * Cross-validate the output metrics.
         */
        @Option(charName = 'x', longName = "cross-validate", usage = "Cross-validate the output metrics.")
        public boolean crossValidation;

        /**
         * The number of cross validation folds.
         */
        @Option(charName = 'n', longName = "num-folds", usage = "The number of cross validation folds.")
        public int numFolds = 5;
    }

    /**
     * @param args the command line arguments
     * @param <T> The {@link Output} subclass.
     */
    @SuppressWarnings("unchecked")
    public static <T extends Output<T>> void main(String[] args) {

        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        ConfigurableTrainTestOptions o = new ConfigurableTrainTestOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.general.trainingPath == null || o.general.testingPath == null || o.outputFactory == null) {
            logger.info(cm.usage());
            System.exit(1);
        }

        Pair<Dataset<T>,Dataset<T>> data = null;
        try {
             data = o.general.load((OutputFactory<T>)o.outputFactory);
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Failed to load data", e);
            System.exit(1);
        }
        Dataset<T> train = data.getA();
        Dataset<T> test = data.getB();

        if (o.trainer == null) {
            logger.warning("No trainer supplied");
            logger.info(cm.usage());
            System.exit(1);
        }

        if (o.transformationMap != null) {
            o.trainer = new TransformTrainer<>(o.trainer,o.transformationMap);
        }
        logger.info("Trainer is " + o.trainer.getProvenance().toString());

        logger.info("Outputs are " + train.getOutputInfo().toReadableString());

        logger.info("Number of features: " + train.getFeatureMap().size());

        final long trainStart = System.currentTimeMillis();
        Model<T> model = ((Trainer<T>)o.trainer).train(train);
        final long trainStop = System.currentTimeMillis();
                
        logger.info("Finished training classifier " + Util.formatDuration(trainStart,trainStop));

        Evaluator<T,? extends Evaluation<T>> evaluator = train.getOutputFactory().getEvaluator();
        final long testStart = System.currentTimeMillis();
        Evaluation<T> evaluation = evaluator.evaluate(model,test);
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart,testStop));
        System.out.println(evaluation.toString());

        if (o.general.outputPath != null) {
            try {
                o.general.saveModel(model);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Error writing model", e);
            }
        }

        if (o.crossValidation) {
            if (o.numFolds > 1) {
                logger.info("Running " + o.numFolds + " fold cross-validation");
                CrossValidation<T,? extends Evaluation<T>> cv = new CrossValidation<>((Trainer<T>)o.trainer,train,evaluator,o.numFolds,o.general.seed);
                List<? extends Pair<? extends Evaluation<T>, Model<T>>> evaluations = cv.evaluate();
                List<Evaluation<T>> evals = evaluations.stream().map(Pair::getA).collect(Collectors.toList());
                // Summarize across everything
                Map<MetricID<T>, DescriptiveStats> summary = EvaluationAggregator.summarize(evals);

                List<MetricID<T>> keys = new ArrayList<>(summary.keySet())
                        .stream()
                        .sorted(Comparator.comparing(Pair::getB))
                        .collect(Collectors.toList());
                System.out.println("Summary across the folds:");
                for (MetricID<T> key : keys) {
                    DescriptiveStats stats = summary.get(key);
                    System.out.printf("%-10s  %.5f (%.5f)%n", key, stats.getMean(), stats.getStandardDeviation());
                }
            } else {
                logger.warning("The number of cross-validation folds must be greater than 1, found " + o.numFolds);
            }
        }
    }
}
