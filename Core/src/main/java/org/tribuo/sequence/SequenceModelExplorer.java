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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.VariableInfo;
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
 * A CLI for interacting with a {@link SequenceModel}.
 */
public class SequenceModelExplorer implements CommandGroup {
    private static final Logger logger = Logger.getLogger(SequenceModelExplorer.class.getName());

    private final CommandInterpreter shell;

    private SequenceModel<?> model;

    /**
     * Builds a sequence model explorer shell.
     */
    public SequenceModelExplorer() {
        shell = new CommandInterpreter();
        shell.setPrompt("model sh% ");
    }

    @Override
    public String getName() {
        return "Sequence Model Explorer";
    }

    @Override
    public String getDescription() {
        return "Commands for inspecting a SequenceModel.";
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
     * @param protobuf Load in a protobuf format model.
     * @return A status string.
     */
    @Command(usage = "<filename> <is-protobuf-format> - Load a model from disk.", completers="fileCompleter")
    public String loadModel(CommandInterpreter ci, File path, boolean protobuf) {
        String output = "Failed to load model";
        if (protobuf) {
            try {
                model = SequenceModel.deserializeFromFile(path.toPath());
                output = "Loaded model from path " + path.getAbsolutePath();
            } catch (IllegalStateException e) {
                logger.log(Level.SEVERE, "Failed to deserialize protobuf when reading from file " + path.getAbsolutePath(), e);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "IOException when reading from file " + path.getAbsolutePath(), e);
            }
        } else {
            try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
                model = (SequenceModel<?>) ois.readObject();
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
     * @param featureName The feature name.
     * @return The feature information.
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
     * Shows the output information.
     * @param ci The command interpreter.
     * @return The output information.
     */
    @Command(usage="Shows the output information.")
    public String outputInfo(CommandInterpreter ci) {
        return model.getOutputIDInfo().toReadableString();
    }

    /**
     * Shows the top n features in this model.
     * @param ci The command interpreter.
     * @param numFeatures The number of features to display.
     * @return The top features.
     */
    @Command(usage="<int> - Shows the top N features in the model")
    public String topFeatures(CommandInterpreter ci, int numFeatures) {
        return ""+ model.getTopFeatures(numFeatures);
    }

    /**
     * Shows the number of features in this model.
     * @param ci The command interpreter.
     * @return The number of features.
     */
    @Command(usage="Shows the number of features in the model")
    public String numFeatures(CommandInterpreter ci) {
        return ""+ model.getFeatureIDMap().size();
    }

    /**
     * Shows the number of features which occurred more than minCount times in the training data.
     * @param ci The command interpreter.
     * @param minCount The minimum occurrence count.
     * @return The number of features which occurred more than minCount times.
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
     * Command line options.
     */
    public static class SequenceModelExplorerOptions implements Options {
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
     * Runs the sequence model explorer.
     * @param args CLI arguments.
     */
    public static void main(String[] args) {
        SequenceModelExplorerOptions options = new SequenceModelExplorerOptions();
        ConfigurationManager cm = new ConfigurationManager(args,options,false);
        SequenceModelExplorer driver = new SequenceModelExplorer();
        if (options.modelFilename != null) {
            logger.log(Level.INFO,driver.loadModel(driver.shell, new File(options.modelFilename), options.protobufFormat));
        }
        driver.startShell();
    }
}
