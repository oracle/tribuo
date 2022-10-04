/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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
package org.tribuo.interop.modelcard;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.Model;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

/**
 * A command line interface for creating and appending UsageDetails to the serialized version of an
 * existing ModelCard.
 */
public class ModelCardCLI implements CommandGroup {
    private static final Logger logger = Logger.getLogger(ModelCardCLI.class.getName());
    /**
     * The command shell instance.
     */
    private final CommandInterpreter shell = new CommandInterpreter();

    /**
     * The model instance.
     */
    private Model<?> model;

    /**
     * The model card.
     */
    private ModelCard modelCard;

    /**
     * The {@link UsageDetailsBuilder} instance.
     */
    private UsageDetailsBuilder builder = new UsageDetailsBuilder();
    private final List<String> outOfScopeUses = new ArrayList<>();
    private final List<String> preProcessingSteps = new ArrayList<>();
    private final List<String> considerations = new ArrayList<>();
    private final List<String> factors = new ArrayList<>();
    private final List<String> resources = new ArrayList<>();

    /**
     * Starts the command shell.
     */
    public void startShell() {
        shell.setPrompt("modelcard sh%% ");
        shell.add(this);
        shell.start();
    }

    @Override
    public String getName() {
        return "ModelCardCLI";
    }

    @Override
    public String getDescription() {
        return "CLI for building a UsageDetails for a model card.";
    }

    /**
     * Loads in a model and builds a partial model card from the provenance.
     * @param ci The command shell.
     * @param path The path to load.
     * @param protobuf Is the model a protobuf?
     * @return A status string.
     */
    @Command(usage = "<path> <is-protobuf> - Loads a model in and builds a partial model card from the provenance.")
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

        modelCard = new ModelCard(model);

        return output;
    }

    /**
     * Records the intended use of the model documented by the ModelCard.
     * @param ci The command shell.
     * @param use The intended use of model.
     * @return A status string.
     */
    @Command(usage = "<String> Records intended use of model.")
    public String intendedUse(CommandInterpreter ci, String use) {
        builder.intendedUse(use);
        return("Recorded intended use as " + use + ".");
    }

    /**
     * Records the intended users of the model documented by the ModelCard.
     * @param ci The command shell.
     * @param users The intended users of model.
     * @return A status string.
     */
    @Command(usage = "<String> Records intended users of model.")
    public String intendedUsers(CommandInterpreter ci, String users) {
        builder.intendedUsers(users);
        return("Recorded intended users as " + users + ".");
    }

    /**
     * Adds an out-of-scope use of the model documented by the ModelCard to its list of out-of-scope uses.
     * @param ci The command shell.
     * @param use The description of an out-of-scope use of the model.
     * @return A status string.
     */
    @Command(usage = "<String> Adds an out-of-scope use to list of out-of-scope uses.")
    public String addOutOfScopeUse(CommandInterpreter ci, String use) {
        outOfScopeUses.add(use);
        return("Added an out-of-scope use to list of out-of-scope uses.");
    }

    /**
     * Removes an out-of-scope use of the model documented by the ModelCard from its list of out-of-scope uses.
     * @param ci The command shell.
     * @param index The index of the out-of-scope use to be removed.
     * @return A status string.
     */
    @Command(usage = "<int> Remove out-of-scope use at specified index (0-indexed).")
    public String removeOutOfScopeUse(CommandInterpreter ci, int index) {
        outOfScopeUses.remove(index);
        return("Removed out-of-scope use at specified index.");
    }

    /**
     * Prints all recorded out-of-scope uses of the model documented by the ModelCard.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays all added out-of-scope uses.")
    public String viewOutOfScopeUse(CommandInterpreter ci) {
        for (int i = 0; i < outOfScopeUses.size(); i++) {
            System.out.println("\t" + i + ") "+ outOfScopeUses.get(i));
        }
        return("Displayed all added out-of-scope uses.");
    }

    /**
     * Adds a pre-processing step for the model documented by the ModelCard to its list of pre-processing steps.
     * @param ci The command shell.
     * @param step The description of a pre-processing step.
     * @return A status string.
     */
    @Command(usage = "<String> Adds pre-processing step to list of steps.")
    public String addPreProcessingStep(CommandInterpreter ci, String step) {
        preProcessingSteps.add(step);
        return("Added pre-processing step to list of steps.");
    }

    /**
     * Removes a pre-processing step of the model documented by the ModelCard from its list of pre-processing steps.
     * @param ci The command shell.
     * @param index The index of the pre-processing step to be removed.
     * @return A status string.
     */
    @Command(usage = "<int> Remove pro-processing step at specified index (0-indexed).")
    public String removePreProcessingStep(CommandInterpreter ci, int index) {
        preProcessingSteps.remove(index);
        return("Removed pre-processing step at specified index.");
    }

    /**
     * Prints all recorded pro-processing of the model documented by the ModelCard.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays all added pre-processing steps.")
    public String viewPreProcessingSteps(CommandInterpreter ci) {
        for (int i = 0; i < preProcessingSteps.size(); i++) {
            System.out.println("\t" + i + ") "+ preProcessingSteps.get(i));
        }
        return("Displayed all added pre-processing steps.");
    }

    /**
     * Adds a consideration for the model documented by the ModelCard to its list of considerations.
     * @param ci The command shell.
     * @param consideration The description of a consideration.
     * @return A status string.
     */
    @Command(usage = "<String> Adds consideration to list of considerations.")
    public String addConsideration(CommandInterpreter ci, String consideration) {
        considerations.add(consideration);
        return("Added consideration to list of considerations.");
    }

    /**
     * Removes a consideration of the model documented by the ModelCard from its list of considerations.
     * @param ci The command shell.
     * @param index The index of the consideration to be removed.
     * @return A status string.
     */
    @Command(usage = "<int> Remove consideration at specified index (0-indexed).")
    public String removeConsideration(CommandInterpreter ci, int index) {
        considerations.remove(index);
        return("Removed consideration at specified index.");
    }

    /**
     * Prints all recorded considerations of the model documented by the ModelCard.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays all added considerations.")
    public String viewConsiderations(CommandInterpreter ci) {
        for (int i = 0; i < considerations.size(); i++) {
            System.out.println("\t" + i + ") "+ considerations.get(i));
        }
        return("Displayed all added considerations.");
    }

    /**
     * Adds a factor for the model documented by the ModelCard to its list of factors.
     * @param ci The command shell.
     * @param factor The description of a factor.
     * @return A status string.
     */
    @Command(usage = "<String> Adds relevant factor to list of factors.")
    public String addFactor(CommandInterpreter ci, String factor) {
        factors.add(factor);
        return("Added factor to list of factors.");
    }

    /**
     * Removes a factor of the model documented by the ModelCard from its list of factors.
     * @param ci The command shell.
     * @param index The index of the factor to be removed.
     * @return A status string.
     */
    @Command(
            usage = "<int> Remove factor at specified index (0-indexed)."
    )
    public String removeFactor(CommandInterpreter ci, int index) {
        factors.remove(index);
        return("Removed factor at specified index.");
    }

    /**
     * Prints all recorded factors of the model documented by the ModelCard.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays all added factors.")
    public String viewFactors(CommandInterpreter ci) {
        for (int i = 0; i < factors.size(); i++) {
            System.out.println("\t" + i + ") "+ factors.get(i));
        }
        return("Displayed all added factors.");
    }

    /**
     * Adds a resource for the model documented by the ModelCard to its list of resources.
     * @param ci The command shell.
     * @param resource The description of a resource.
     * @return A status string.
     */
    @Command(usage = "<String> Adds resource to list of resources.")
    public String addResource(CommandInterpreter ci, String resource) {
        resources.add(resource);
        return("Added resource to list of resources.");
    }

    /**
     * Removes a resource of the model documented by the ModelCard from its list of resources.
     * @param ci The command shell.
     * @param index The index of the resource to be removed.
     * @return A status string.
     */
    @Command(usage = "<int> Remove resource at specified index (0-indexed).")
    public String removeResource(CommandInterpreter ci, int index) {
        resources.remove(index);
        return("Removed resource at specified index.");
    }

    /**
     * Prints all recorded resources of the model documented by the ModelCard.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays all added resources.")
    public String viewResources(CommandInterpreter ci) {
        for (int i = 0; i < resources.size(); i++) {
            System.out.println("\t" + i + ") "+ resources.get(i));
        }
        return("Displayed all added resources.");
    }

    /**
     * Records the primary contact person of the model documented by the ModelCard.
     * @param ci The command shell.
     * @param contact The primary contact person of the model.
     * @return A status string.
     */
    @Command(usage = "<String> Records primary contact in case of questions or comments.")
    public String primaryContact(CommandInterpreter ci, String contact) {
        builder.primaryContact(contact);
        return("Recorded primary contact as " + contact + ".");
    }

    /**
     * Records the citation the model documented by the ModelCard.
     * @param ci The command shell.
     * @param citation The citation the model.
     * @return A status string.
     */
    @Command(usage = "<String> Records model's citation.")
    public String modelCitation(CommandInterpreter ci, String citation) {
        builder.modelCitation(citation);
        return("Recorded model citation as " + citation + ".");
    }

    /**
     * Records the license the model documented by the ModelCard.
     * @param ci The command shell.
     * @param license The license the model.
     * @return A status string.
     */
    @Command(usage = "<String> Records model's license.")
    public String modelLicense(CommandInterpreter ci, String license) {
        builder.modelLicense(license);
        return("Recorded model license as " + license + ".");
    }

    /**
     * Creates an instance of {@link UsageDetails} using the fields recorded by the builder.
     * @return An instance of {@link UsageDetails}.
     */
    private UsageDetails createUsageDetails() {
        builder.outOfScopeUses(outOfScopeUses);
        builder.preProcessingSteps(preProcessingSteps);
        builder.considerations(considerations);
        builder.factors(factors);
        builder.resources(resources);
        return builder.build();
    }

    /**
     * Loads a model card in from disk. Fails if the model card already includes the usage information.
     * @param ci The command interpreter.
     * @param modelCardFile The model card file to load.
     * @return A status string.
     * @throws IOException If the card failed to load.
     */
    @Command(usage = "<filename> - Loads in an existing model card without usage details")
    public String loadModelCard(CommandInterpreter ci, File modelCardFile) throws IOException {
        var modelCardTmp = ModelCard.deserializeFromJson(modelCardFile.toPath());
        if (modelCardTmp.getUsageDetails() != null) {
            return "This ModelCard already contains a UsageDetails.";
        } else {
            modelCard = modelCardTmp;
            return "Loaded model card from " + modelCardFile;
        }
    }

    /**
     * Saves a serialized version of the {@link UsageDetails} created by the builder to the destination file.
     * <p>
     * Throws {@link IllegalArgumentException} if the serialized ModelCard stored at the destination file already
     * contains a non-null UsageDetails.
     * @param ci The command shelll.
     * @param destinationFile The file path where the model card will be saved in json format.
     * @return A status string.
     * @throws IOException if a problem is encountered when writing to the file.
     */
    @Command(usage = "<filename> Saves UsageDetails to an existing ModelCard file.")
    public String saveUsageDetails(CommandInterpreter ci, File destinationFile) throws IOException {
        if (modelCard == null) {
            return "Error, must load in a model card before saving the completed one.";
        }

        UsageDetails usageDetails = createUsageDetails();

        ObjectNode usageDetailsObject = usageDetails.toJson();

        ObjectNode modelCardObject = modelCard.toJson();
        modelCardObject.set("UsageDetails", usageDetailsObject);
        mapper.writeValue(destinationFile, modelCardObject);

        return "Saved UsageDetails to destination file '" + destinationFile + "'.";
    }

    /**
     * Creates a new instance of {@link UsageDetailsBuilder} to allow a new {@link UsageDetails} to be written.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Removes all previously written fields for UsageDetails and clears the model to write a new UsageDetails.")
    public String clearUsageDetails(CommandInterpreter ci) {
        builder = new UsageDetailsBuilder();
        outOfScopeUses.clear();
        preProcessingSteps.clear();
        considerations.clear();
        factors.clear();
        resources.clear();
        model = null;
        modelCard = null;
        return "Started a new UsageDetails.";
    }

    /**
     * Prints the fields recorded by the user for their {@link UsageDetails} object.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays current state of UsageDetails.")
    public String viewUsageDetails(CommandInterpreter ci) {
        System.out.println(createUsageDetails());
        return "Displayed current state of UsageDetails.";
    }

    /**
     * Prints the current model card, incorporating the current usage details.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays the current model card.")
    public String viewModelCard(CommandInterpreter ci) {
        if (modelCard == null) {
            return "Error, must load in a model card before viewing it.";
        }

        UsageDetails usageDetails = createUsageDetails();

        ObjectNode usageDetailsObject = usageDetails.toJson();

        ObjectNode modelCardObject = modelCard.toJson();
        modelCardObject.set("UsageDetails", usageDetailsObject);

        return modelCardObject.toPrettyString();
    }

    /**
     * Prints the current model provenance.
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Displays the currently loaded model, if there is one.")
    public String viewModelProvenance(CommandInterpreter ci) {
        if (model == null) {
            return "Error, must load a model to view it.";
        }
        try {
            System.out.println(mapper.writeValueAsString(ProvenanceUtil.convertToMap(model.getProvenance())));
        } catch (JsonProcessingException e) {
            logger.log(Level.SEVERE, "Failed to produce JSON from model provenance", e);
        }
        return "";
    }

    /**
     * Closes the command shell
     * @param ci The command shell.
     * @return A status string.
     */
    @Command(usage = "Closes CLI without explicitly saving anything recorded.")
    public String close(CommandInterpreter ci) {
        shell.close();
        return "Closed ClI.";
    }

    /**
     * CLI options for {@link ModelCardCLI}.
     */
    public static class ModelCardCLIOptions implements Options {
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
        ModelCardCLIOptions options = new ModelCardCLIOptions();
        ConfigurationManager cm = new ConfigurationManager(args,options,false);
        ModelCardCLI driver = new ModelCardCLI();
        if (options.modelFilename != null) {
            logger.log(Level.INFO,driver.loadModel(driver.shell, new File(options.modelFilename), options.protobufFormat));
        }
        driver.startShell();
    }
}
