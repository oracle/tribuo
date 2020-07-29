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

    protected CommandInterpreter shell;

    private SequenceModel<?> model;

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

    @Command(usage = "<filename> - Load a model from disk.", completers="fileCompleter")
    public String loadModel(CommandInterpreter ci, File path) {
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(path)))) {
            model = (SequenceModel<?>) ois.readObject();
        } catch (ClassNotFoundException e) {
            logger.log(Level.SEVERE,"Failed to load class from stream " + path.getAbsolutePath(),e);
            return "Failed to load model";
        } catch (FileNotFoundException e) {
            logger.log(Level.SEVERE,"Failed to open file " + path.getAbsolutePath(),e);
            return "Failed to load model";
        } catch (IOException e) {
            logger.log(Level.SEVERE,"IOException when reading from file " + path.getAbsolutePath(),e);
            return "Failed to load model";
        }

        return "Loaded model from path " + path.toString();
    }

    @Command(usage="Shows the model description")
    public String modelDescription(CommandInterpreter ci) {
        return model.toString();
    }

    @Command(usage="Shows the information on a particular feature")
    public String featureInfo(CommandInterpreter ci, String featureName) {
        VariableInfo f = model.getFeatureIDMap().get(featureName);
        if (f != null) {
            return "" + f.toString();
        } else {
            return "Feature " + featureName + " not found.";
        }
    }

    @Command(usage="Shows the output information.")
    public String outputInfo(CommandInterpreter ci) {
        return model.getOutputIDInfo().toReadableString();
    }

    @Command(usage="<int> - Shows the top N features in the model")
    public String topFeatures(CommandInterpreter ci, int numFeatures) {
        return ""+ model.getTopFeatures(numFeatures);
    }

    @Command(usage="Shows the number of features in the model")
    public String numFeatures(CommandInterpreter ci) {
        return ""+ model.getFeatureIDMap().size();
    }

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

    public static String usage() {
        StringBuilder string = new StringBuilder();
        string.append("Usage: ModelExplorer\n");

        string.append("Optional parameters\n");
        string.append("     -f <model-filename>\n");
        string.append("         Load in a model from file.\n");

        return string.toString();
    }

    public static class SequenceModelExplorerOptions implements Options {
        @Option(charName='f',longName="filename",usage="Model file to load. Optional.")
        public String modelFilename;
    }

    public static void main(String[] args) {
        SequenceModelExplorerOptions options = new SequenceModelExplorerOptions();
        ConfigurationManager cm = new ConfigurationManager(args,options,false);
        SequenceModelExplorer driver = new SequenceModelExplorer();
        if (options.modelFilename != null) {
            logger.log(Level.INFO,driver.loadModel(driver.shell, new File(options.modelFilename)));
        }
        driver.startShell();
    }
}
