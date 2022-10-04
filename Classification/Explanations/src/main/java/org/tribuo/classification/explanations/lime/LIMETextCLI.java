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

package org.tribuo.classification.explanations.lime;

import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.SparseModel;
import org.tribuo.SparseTrainer;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.data.text.impl.BasicPipeline;
import org.tribuo.data.text.impl.TextFeatureExtractorImpl;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.rtree.CARTJointRegressionTrainer;
import org.jline.builtins.Completers;
import org.jline.reader.Completer;
import org.jline.reader.impl.completer.NullCompleter;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.universal.UniversalTokenizer;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.SplittableRandom;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A CLI for interacting with {@link LIMEText}. Uses a simple tokenisation and text extraction pipeline.
 */
public class LIMETextCLI implements CommandGroup {
    private static final Logger logger = Logger.getLogger(LIMETextCLI.class.getName());

    private final CommandInterpreter shell;

    private Model<Label> model;

    private int numSamples = 100;

    private int numFeatures = 10;

    //private SparseTrainer<Regressor> limeTrainer = new LARSLassoTrainer(numFeatures);
    private SparseTrainer<Regressor> limeTrainer = new CARTJointRegressionTrainer((int)Math.log(numFeatures),true);

    private Tokenizer tokenizer = new UniversalTokenizer();

    private TextFeatureExtractor<Label> extractor = new TextFeatureExtractorImpl<>(new BasicPipeline(tokenizer,2));

    private LIMEText limeText = null;

    /**
     * Constructs a LIME CLI.
     */
    public LIMETextCLI() {
        shell = new CommandInterpreter();
        shell.setPrompt("lime-text sh% ");
    }

    @Override
    public String getName() {
        return "LIME Text CLI";
    }

    @Override
    public String getDescription() {
        return "Commands for experimenting with LIME Text.";
    }

    /**
     * Completers for filenames.
     * @return The filename completers.
     */
    public Completer[] fileCompleter() {
        return new Completer[]{
                new Completers.FileNameCompleter(),
                new NullCompleter()
        };
    }

    /**
     * Start the command shell
     */
    public void startShell() {
        shell.add(this);
        shell.start();
    }

    /**
     * Loads a model in from disk.
     * @param ci The command interpreter.
     * @param path The path to load the model from.
     * @param protobuf Load the model from protobuf?
     * @return A status message.
     */
    @Command(usage = "<filename> <load-protobuf> - Load a model from disk.", completers="fileCompleter")
    public String loadModel(CommandInterpreter ci, File path, boolean protobuf) {
        String output = "Failed to load model";
        if (protobuf) {
            try {
                Model<?> tmpModel = Model.deserializeFromFile(path.toPath());
                model = tmpModel.castModel(Label.class);
                output = "Loaded model from path " + path.getAbsolutePath();
            } catch (IllegalStateException e) {
                logger.log(Level.SEVERE, "Failed to deserialize protobuf when reading from file " + path.getAbsolutePath(), e);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "IOException when reading from file " + path.getAbsolutePath(), e);
            }
        } else {
            try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
                Model<?> tmpModel = (Model<?>) ois.readObject();
                model = tmpModel.castModel(Label.class);
                output = "Loaded model from path " + path.getAbsolutePath();
            } catch (ClassNotFoundException e) {
                logger.log(Level.SEVERE, "Failed to load class from stream " + path.getAbsolutePath(), e);
            } catch (FileNotFoundException e) {
                logger.log(Level.SEVERE, "Failed to open file " + path.getAbsolutePath(), e);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "IOException when reading from file " + path.getAbsolutePath(), e);
            }
        }

        limeText = new LIMEText(new SplittableRandom(1),model,limeTrainer,numSamples,extractor,tokenizer);

        return output;
    }

    /**
     * Does the model generate probabilities.
     * @param ci The command interpreter.
     * @return True if the model generates probabilities.
     */
    @Command(usage="Does the model generate probabilities")
    public String generatesProbabilities(CommandInterpreter ci) {
        return ""+model.generatesProbabilities();
    }

    /**
     * Shows the model description.
     * @param ci The command interpreter.
     * @return The model description.
     */
    @Command(usage="Shows the model description")
    public String modelDescription(CommandInterpreter ci) {
        return model.toString();
    }

    /**
     * Shows information on a particular feature.
     * @param ci The command interpreter.
     * @param featureName The feature to show.
     * @return Feature information.
     */
    @Command(usage="Shows the information on a particular feature")
    public String featureInfo(CommandInterpreter ci, String featureName) {
        VariableInfo f = model.getFeatureIDMap().get(featureName);
        if (f != null) {
            return "" + f.toString();
        } else {
            return "Feature " + featureName + " not found.";
        }
    }

    /**
     * Shows the top features of the loaded model.
     * @param ci The command interpeter.
     * @param numFeatures The number of features to show.
     * @return The top features of the model.
     */
    @Command(usage="<int> - Shows the top N features in the model")
    public String topFeatures(CommandInterpreter ci, int numFeatures) {
        return ""+ model.getTopFeatures(numFeatures);
    }

    /**
     * Shows the number of features.
     * @param ci The command interpreter.
     * @return The number of features in the model.
     */
    @Command(usage="Shows the number of features in the model")
    public String numFeatures(CommandInterpreter ci) {
        return ""+ model.getFeatureIDMap().size();
    }

    /**
     * Shows the number of features that occurred more than minCount times.
     * @param ci The command interpreter.
     * @param minCount The minimum feature occurrence.
     * @return The number of features with more than minCount occurrences.
     */
    @Command(usage="<min count> - Shows the number of features that occurred more than min count times.")
    public String minCount(CommandInterpreter ci, int minCount) {
        int counter = 0;
        for (VariableInfo f : model.getFeatureIDMap()) {
            if (f.getCount() > minCount) {
                counter++;
            }
        }
        return counter + " features occurred more than " + minCount + " times.";
    }

    /**
     * Shows the output statistics.
     * @param ci The command interpreter.
     * @return The output statistics.
     */
    @Command(usage="Shows the output statistics")
    public String showLabelStats(CommandInterpreter ci) {
        return "Label histogram : \n" + model.getOutputIDInfo().toReadableString();
    }

    /**
     * Sets the number of samples to use in LIME.
     * @param ci The command interpreter.
     * @param newNumSamples The number of samples to use in LIME.
     * @return A status message.
     */
    @Command(usage="Sets the number of samples to use in LIME")
    public String setNumSamples(CommandInterpreter ci, int newNumSamples) {
        numSamples = newNumSamples;
        return "Set number of samples to " + numSamples;
    }

    /**
     * Explains a text classification.
     * @param ci The command interpreter.
     * @param tokens A space separated token stream.
     * @return An explanation.
     */
    @Command(usage="Explain a text classification")
    public String explain(CommandInterpreter ci, String[] tokens) {
        String text = String.join(" ",tokens);

        LIMEExplanation explanation = limeText.explain(text);

        SparseModel<Regressor> model = explanation.getModel();

        ci.out.println("Active features of the predicted class = " + model.getActiveFeatures().get(explanation.getPrediction().getOutput().getLabel()));

        return "Explanation = " + explanation.toString();
    }

    /**
     * Sets the number of features LIME should use in an explanation.
     * @param ci The command interpreter.
     * @param newNumFeatures The number of features.
     * @return A status message.
     */
    @Command(usage="Sets the number of features LIME should use in an explanation")
    public String setNumFeatures(CommandInterpreter ci, int newNumFeatures) {
        numFeatures = newNumFeatures;
        //limeTrainer = new LARSLassoTrainer(numFeatures);
        limeTrainer = new CARTJointRegressionTrainer((int)Math.log(numFeatures),true);
        limeText = new LIMEText(new SplittableRandom(1),model,limeTrainer,numSamples,extractor, tokenizer);
        return "Set the number of features in LIME to " + numFeatures;
    }

    /**
     * Makes a prediction using the loaded model.
     * @param ci The command interpreter.
     * @param tokens A space separated token stream.
     * @return The prediction.
     */
    @Command(usage="Make a prediction")
    public String predict(CommandInterpreter ci, String[] tokens) {
        String text = String.join(" ",tokens);

        Prediction<Label> prediction = model.predict(extractor.extract(LabelFactory.UNKNOWN_LABEL,text));

        return "Prediction = " + prediction.toString();
    }

    /**
     * Command line options.
     */
    public static class LIMETextCLIOptions implements Options {
        /**
         * Model file to load. Optional.
         */
        @Option(charName = 'f', longName = "filename", usage = "Model file to load. Optional.")
        public String modelFilename;

        /**
         * Load the model from a protobuf. Optional.
         */
        @Option(charName = 'p', longName = "protobuf-model", usage = "Load the model from a protobuf. Optional")
        public boolean protobufFormat;
    }

    /**
     * Runs a LIMETextCLI.
     * @param args The CLI arguments.
     */
    public static void main(String[] args) {
        LIMETextCLI.LIMETextCLIOptions options = new LIMETextCLI.LIMETextCLIOptions();
        try {
            ConfigurationManager cm = new ConfigurationManager(args, options, false);
            LIMETextCLI driver = new LIMETextCLI();
            if (options.modelFilename != null) {
                logger.log(Level.INFO, driver.loadModel(driver.shell, new File(options.modelFilename), options.protobufFormat));
            }
            driver.startShell();
        } catch (UsageException e) {
            System.out.println("Usage: " + e.getUsage());
        }
    }
}
