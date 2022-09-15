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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.jline.builtins.Completers;
import org.jline.reader.Completer;
import org.jline.reader.impl.completer.NullCompleter;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A command line interface for loading in models and inspecting their feature and output spaces.
 */
public class ModelExplorer implements CommandGroup {
    private static final Logger logger = Logger.getLogger(ModelExplorer.class.getName());

    /**
     * The command shell instance.
     */
    private final CommandInterpreter shell;

    private Model<?> model;

    /**
     * Builds a new model explorer shell.
     */
    public ModelExplorer() {
        shell = new CommandInterpreter();
        shell.setPrompt("model sh% ");
    }

    @Override
    public String getName() {
        return "Model Explorer";
    }

    @Override
    public String getDescription() {
        return "Commands for inspecting a Model.";
    }

    /**
     * Completers for files.
     * @return The completers for file commands.
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
     * Loads a model.
     * @param ci The shell instance.
     * @param path The path to load.
     * @param protobuf If the model is a protobuf.
     * @return A status string.
     */
    @Command(usage = "<filename> - Load a model from disk.", completers="fileCompleter")
    public String loadModel(CommandInterpreter ci, File path, boolean protobuf) {
        String output = "Failed to load model";
        if (protobuf) {
            try {
                model = Model.deserializeFromFile(path.toPath());
                output = "Loaded model from path " + path.getAbsolutePath();
            } catch (IllegalStateException e) {
                logger.log(Level.SEVERE, "Failed to deserialize protobuf when reading from file " + path.getAbsolutePath(), e);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "IOException when reading from file " + path.getAbsolutePath(), e);
            }
        } else {
            try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
                model = (Model<?>) ois.readObject();
                output = "Loaded model from path " + path.getAbsolutePath();
            } catch (ClassNotFoundException e) {
                logger.log(Level.SEVERE, "Failed to load class from stream " + path.getAbsolutePath(), e);
            } catch (FileNotFoundException e) {
                logger.log(Level.SEVERE, "Failed to open file " + path.getAbsolutePath(), e);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "IOException when reading from file " + path.getAbsolutePath(), e);
            }
        }

        return output;
    }

    /**
     * Checks if the model generates probabilities.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage="Does the model generate probabilities")
    public String generatesProbabilities(CommandInterpreter ci) {
        return ""+model.generatesProbabilities();
    }

    /**
     * Displays the model provenance.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage="Shows the model provenance")
    public String modelProvenance(CommandInterpreter ci) {
        return model.getProvenance().toString();
    }

    /**
     * Shows a specific feature's information.
     * @param ci The command shell.
     * @param featureName The feature name.
     * @return A status string.
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
     * Displays the output info.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage="Shows the output information.")
    public String outputInfo(CommandInterpreter ci) {
        return model.getOutputIDInfo().toReadableString();
    }

    /**
     * Displays the top n features.
     * @param ci The command shell
     * @param numFeatures The number of features to display.
     * @return A status string.
     */
    @Command(usage="<int> - Shows the top N features in the model")
    public String topFeatures(CommandInterpreter ci, int numFeatures) {
        return ""+ model.getTopFeatures(numFeatures);
    }

    /**
     * Displays the number of features.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage="Shows the number of features in the model")
    public String numFeatures(CommandInterpreter ci) {
        return ""+ model.getFeatureIDMap().size();
    }

    /**
     * Shows the number of features which occurred more than min count times.
     * @param ci The command shell.
     * @param minCount The minimum occurrence count.
     * @return A status string.
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
     * CLI options for {@link ModelExplorer}.
     */
    public static class ModelExplorerOptions implements Options {
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
     * Entry point.
     * @param args CLI args.
     */
    public static void main(String[] args) {
        ModelExplorerOptions options = new ModelExplorerOptions();
        ConfigurationManager cm = new ConfigurationManager(args,options,false);
        ModelExplorer driver = new ModelExplorer();
        if (options.modelFilename != null) {
            logger.log(Level.INFO,driver.loadModel(driver.shell, new File(options.modelFilename), options.protobufFormat));
        }
        driver.startShell();
    }
}
