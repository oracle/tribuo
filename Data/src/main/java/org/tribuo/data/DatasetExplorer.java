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

import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Dataset;
import org.tribuo.VariableInfo;
import org.tribuo.data.csv.CSVSaver;
import org.jline.builtins.Completers;
import org.jline.reader.Completer;
import org.jline.reader.impl.completer.NullCompleter;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A CLI for exploring a serialised {@link Dataset}.
 */
public final class DatasetExplorer implements CommandGroup {
    private static final Logger logger = Logger.getLogger(DatasetExplorer.class.getName());

    private final CommandInterpreter shell;

    private Dataset<?> dataset;

    /**
     * Constructs a dataset explorer.
     */
    public DatasetExplorer() {
        shell = new CommandInterpreter();
        shell.setPrompt("dataset sh% ");
    }

    @Override
    public String getName() {
        return "Dataset Explorer";
    }

    @Override
    public String getDescription() {
        return "Commands for inspecting a Dataset.";
    }

    /**
     * The filename completer.
     * @return The completer array.
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
     * Loads a serialized dataset.
     * @param ci The command interpreter.
     * @param path The path to load.
     * @param protobuf Load the model from protobuf?
     * @return A status string.
     */
    @Command(usage = "<filename> <is-protobuf> - Load a dataset from disk.", completers="fileCompleter")
    public String loadDataset(CommandInterpreter ci, File path, boolean protobuf) {
        String output = "Failed to load dataset";
        if (protobuf) {
            try {
                dataset = Dataset.deserializeFromFile(path.toPath());
                output = "Loaded dataset from path " + path.getAbsolutePath();
            } catch (IllegalStateException e) {
                logger.log(Level.SEVERE, "Failed to deserialize protobuf when reading from file " + path.getAbsolutePath(), e);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "IOException when reading from file " + path.getAbsolutePath(), e);
            }
        } else {
            try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
                dataset = (Dataset<?>) ois.readObject();
                output = "Loaded dataset from path " + path.getAbsolutePath();
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
     * Shows information on a particular feature.
     * @param ci The command interpreter.
     * @param featureName The feature name.
     * @return The feature information.
     */
    @Command(usage="Shows the information on a particular feature")
    public String featureInfo(CommandInterpreter ci, String featureName) {
        VariableInfo f = dataset.getFeatureMap().get(featureName);
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
        return dataset.getOutputInfo().toReadableString();
    }

    /**
     * Shows the number of examples in this dataset.
     * @param ci The command interpreter.
     * @return The number of examples.
     */
    @Command(usage="Shows the number of rows in the dataset")
    public String numExamples(CommandInterpreter ci) {
        return ""+dataset.getData().size();
    }

    /**
     * Shows the number of features in this dataset.
     * @param ci The command interpreter.
     * @return The number of features.
     */
    @Command(usage="Shows the number of features in the dataset")
    public String numFeatures(CommandInterpreter ci) {
        return ""+dataset.getFeatureMap().size();
    }

    /**
     * Shows the number of features which occurred more than minCount times in the dataset.
     * @param ci The command interpreter.
     * @param minCount The minimum occurrence count.
     * @return The number of features which occurred more than minCount times.
     */
    @Command(usage="<min count> - Shows the number of features that occurred more than min count times.")
    public String minCount(CommandInterpreter ci, int minCount) {
        int counter = 0;
        for (VariableInfo f : dataset.getFeatureMap()) {
            if (f.getCount() > minCount) {
                counter++;
            }
        }
        return counter + " features occurred more than " + minCount + " times.";
    }

    /**
     * Shows the output statistics.
     * @param ci The command interpreter.
     * @return The output statistics string.
     */
    @Command(usage="Shows the output statistics")
    public String showOutputStats(CommandInterpreter ci) {
        return "Output statistics: \n" + dataset.getOutputInfo().toReadableString();
    }

    /**
     * Saves out the dataset as a CSV file.
     * @param ci The command interpreter.
     * @param path The path to save to.
     * @return A status message.
     */
    @Command(usage="Saves out the data as a CSV.")
    public String saveCSV(CommandInterpreter ci, String path) {
        CSVSaver saver = new CSVSaver();
        try {
            saver.save(Paths.get(path),dataset,CSVSaver.DEFAULT_RESPONSE);
            return "Saved";
        } catch (IOException e) {
            e.printStackTrace(ci.out);
            return "Failed to save to CSV.";
        }
    }

    /**
     * Shows the dataset provenance.
     * @param ci The command interpreter.
     * @return The dataset provenance string.
     */
    @Command(usage="Shows the dataset provenance")
    public String showProvenance(CommandInterpreter ci) {
        return dataset.getProvenance().toString();
    }

    /**
     * Command line options.
     */
    public static class DatasetExplorerOptions implements Options {
        /**
         * Dataset file to load. Optional.
         */
        @Option(charName = 'f', longName = "filename", usage = "Dataset file to load. Optional.")
        public String modelFilename;

        /**
         * Load the model from a protobuf. Optional.
         */
        @Option(charName = 'p', longName = "protobuf-model", usage = "Load the model from a protobuf. Optional")
        public boolean protobufFormat;
    }

    /**
     * Runs a dataset explorer.
     * @param args CLI arguments.
     */
    public static void main(String[] args) {
        DatasetExplorerOptions options = new DatasetExplorerOptions();
        ConfigurationManager cm = new ConfigurationManager(args,options,false);
        DatasetExplorer driver = new DatasetExplorer();
        if (options.modelFilename != null) {
            logger.log(Level.INFO,driver.loadDataset(driver.shell, new File(options.modelFilename), options.protobufFormat));
        }
        driver.startShell();
    }
}
