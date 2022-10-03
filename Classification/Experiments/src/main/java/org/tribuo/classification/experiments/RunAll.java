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
import org.tribuo.Model;
import org.tribuo.Trainer;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.ConfusionMatrix;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.data.DataOptions;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Trains and tests a model using the supplied data, for each trainer inside a configuration file.
 */
public class RunAll {
    private static final Logger logger = Logger.getLogger(RunAll.class.getName());

    /**
     * Command line options.
     */
    public static class RunAllOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Performs the same training and test experiment on all Trainers in the supplied configuration file.";
        }

        /**
         * Options for loading in data.
         */
        public DataOptions general;

        /**
         * Directory to write out the models and test reports.
         */
        @Option(charName = 'd', longName = "output-directory", usage = "Directory to write out the models and test reports.")
        public File directory;

        /**
         * Write out models in protobuf format.
         */
        @Option(longName = "write-protobuf-models", usage = "Write out models in protobuf format.")
        public boolean protobuf;
    }

    /**
     * Runs the RunALL CLI.
     * @param args The CLI arguments.
     * @throws IOException If it failed to load the data.
     */
    public static void main(String[] args) throws IOException {
        LabsLogFormatter.setAllLogFormatters();

        RunAllOptions o = new RunAllOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if (o.general.trainingPath == null || o.general.testingPath == null || o.directory == null) {
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

        logger.info("Creating directory - " + o.directory.toString());
        if (!o.directory.exists() && !o.directory.mkdirs()) {
            logger.warning("Failed to create directory.");
        }

        Map<String,Double> performances = new HashMap<>();
        List<Trainer> trainers = cm.lookupAll(Trainer.class);
        for (Trainer<?> t : trainers) {
            String name = t.getClass().getSimpleName();
            logger.info("Training model using " + t.toString());
            @SuppressWarnings("unchecked") // configuration system cast.
            Model<Label> curModel = ((Trainer<Label>)t).train(train);
            LabelEvaluator evaluator = new LabelEvaluator();
            LabelEvaluation evaluation = evaluator.evaluate(curModel,test);
            Double old = performances.put(name,evaluation.microAveragedF1());
            if (old != null) {
                logger.info("Found two trainers with the name " + name);
            }
            String outputPath = o.directory.toString()+"/"+name;
            if (o.protobuf) {
                curModel.serializeToFile(Paths.get(outputPath + ".model"));
            } else {
                try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(outputPath + ".model"))) {
                    oos.writeObject(curModel);
                }
            }
            try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputPath+".output"), StandardCharsets.UTF_8))) {
                writer.println("Model = " + name);
                writer.println("Provenance = " + curModel.toString());
                writer.println();
                ConfusionMatrix<Label> matrix = evaluation.getConfusionMatrix();
                writer.println("ConfusionMatrix:\n" + matrix.toString());
                writer.println();
                writer.println("Evaluation:\n" + evaluation.toString());
            }
        }

        for (Map.Entry<String,Double> e : performances.entrySet()) {
            logger.info("Trainer = " + e.getKey() + ", F1 = " + e.getValue());
        }

    }

}
