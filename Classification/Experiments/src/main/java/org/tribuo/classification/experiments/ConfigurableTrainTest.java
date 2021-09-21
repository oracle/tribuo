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

package org.tribuo.classification.experiments;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.MutableDataset;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.WeightedExamples;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.WeightedLabels;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.data.DataOptions;
import org.tribuo.util.Util;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Build and run a classifier for a standard dataset.
 */
public class ConfigurableTrainTest {

    private static final Logger logger = Logger.getLogger(ConfigurableTrainTest.class.getName());

    /**
     * Command line options.
     */
    public static class ConfigurableTrainTestOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Loads a Trainer (and optionally a Datasource) from a config file, trains a Model, tests it and optionally saves it to disk.";
        }

        /**
         * Options for loading in data.
         */
        public DataOptions general;

        /**
         * Load a trainer from the config file.
         */
        @Option(charName = 't', longName = "trainer", usage = "Load a trainer from the config file.")
        public Trainer<Label> trainer;

        /**
         * A list of weights to use in classification. Format = LABEL_NAME:weight,LABEL_NAME:weight...
         */
        @Option(charName = 'w', longName = "weights", usage = "A list of weights to use in classification. Format = LABEL_NAME:weight,LABEL_NAME:weight...")
        public List<String> weights;

        /**
         * Path to write model predictions
         */
        @Option(charName = 'o', longName = "predictions", usage = "Path to write model predictions")
        public Path predictionPath;
    }

    /**
     * Converts the weight text input format into an object suitable for use in a Trainer.
     * @param input The input form.
     * @return The weights.
     */
    public static Map<Label,Float> processWeights(List<String> input) {
        Map<Label,Float> map = new HashMap<>();

        for (String tuple : input) {
            String[] splitTuple = tuple.split(":");
            map.put(new Label(splitTuple[0]),Float.parseFloat(splitTuple[1]));
        }

        return map;
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

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

        if (o.general.trainingPath == null || o.general.testingPath == null) {
            logger.info(cm.usage());
            System.exit(1);
        }
        Pair<Dataset<Label>,Dataset<Label>> data = null;
        try {
             data = o.general.load(new LabelFactory());
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Failed to load data", e);
            System.exit(1);
        }
        Dataset<Label> train = data.getA();
        Dataset<Label> test = data.getB();

        if (o.trainer == null) {
            logger.warning("No trainer supplied");
            logger.info(cm.usage());
            System.exit(1);
        }
        logger.info("Trainer is " + o.trainer.toString());

        if (o.weights != null) {
            Map<Label,Float> weightsMap = processWeights(o.weights);
            if (o.trainer instanceof WeightedLabels) {
                ((WeightedLabels) o.trainer).setLabelWeights(weightsMap);
                logger.info("Setting label weights using " + weightsMap.toString());
            } else if (o.trainer instanceof WeightedExamples) {
                ((MutableDataset<Label>)train).setWeights(weightsMap);
                logger.info("Setting example weights using " + weightsMap.toString());
            } else {
                logger.warning("The selected trainer does not support weighted training. The chosen trainer is " + o.trainer.toString());
                logger.info(cm.usage());
                System.exit(1);
            }
        }

        logger.info("Labels are " + train.getOutputInfo().toReadableString());

        final long trainStart = System.currentTimeMillis();
        Model<Label> model = o.trainer.train(train);
        final long trainStop = System.currentTimeMillis();
                
        logger.info("Finished training classifier " + Util.formatDuration(trainStart,trainStop));

        LabelEvaluator labelEvaluator = new LabelEvaluator();
        final long testStart = System.currentTimeMillis();
        List<Prediction<Label>> predictions = model.predict(test);
        LabelEvaluation labelEvaluation = labelEvaluator.evaluate(model,predictions,test.getProvenance());
        final long testStop = System.currentTimeMillis();
        logger.info("Finished evaluating model " + Util.formatDuration(testStart,testStop));
        System.out.println(labelEvaluation.toString());
        ConfusionMatrix<Label> matrix = labelEvaluation.getConfusionMatrix();
        System.out.println(matrix.toString());
        if (model.generatesProbabilities()) {
            System.out.println("Average AUC = " + labelEvaluation.averageAUCROC(false));
            System.out.println("Average weighted AUC = " + labelEvaluation.averageAUCROC(true));
        }

        if(o.predictionPath!=null) {
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

        if (o.general.outputPath != null) {
            try {
                o.general.saveModel(model);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Error writing model", e);
            }
        }
    }
}
