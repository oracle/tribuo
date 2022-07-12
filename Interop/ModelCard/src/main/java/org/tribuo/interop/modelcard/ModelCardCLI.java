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

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

public class ModelCardCLI implements CommandGroup {
    private final CommandInterpreter shell = new CommandInterpreter();
    private UsageDetailsBuilder builder = new UsageDetailsBuilder();
    private final List<String> outOfScopeUses = new ArrayList<>();
    private final List<String> preProcessingSteps = new ArrayList<>();
    private final List<String> considerations = new ArrayList<>();
    private final List<String> factors = new ArrayList<>();
    private final List<String> resources = new ArrayList<>();

    public void startShell() {
        shell.setPrompt("CLI% ");
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

    @Command(
            usage = "<String> Records intended use of model."
    )
    public String intendedUse(CommandInterpreter ci, String use) {
        builder.intendedUse(use);
        return("Recorded intended use as " + use + ".");
    }

    @Command(
            usage = "<String> Records intended users of model."
    )
    public String intendedUsers(CommandInterpreter ci, String users) {
        builder.intendedUsers(users);
        return("Recorded intended users as " + users + ".");
    }

    @Command(
            usage = "<String> Adds an out-of-scope use to list of out-of-scope uses."
    )
    public String addOutOfScopeUse(CommandInterpreter ci, String use) {
        outOfScopeUses.add(use);
        return("Added an out-of-scope use to list of out-of-scope uses.");
    }

    @Command(
            usage = "<int> Remove out-of-scope use at specified index (0-indexed)."
    )
    public String removeOutOfScopeUse(CommandInterpreter ci, int index) {
        outOfScopeUses.remove(index);
        return("Removed out-of-scope use at specified index.");
    }

    @Command(
            usage = "Displays all added out-of-scope uses."
    )
    public String viewOutOfScopeUse(CommandInterpreter ci) {
        for (int i = 0; i < outOfScopeUses.size(); i++) {
            System.out.println("\t" + i + ") "+ outOfScopeUses.get(i));
        }
        return("Displayed all added out-of-scope uses.");
    }

    @Command(
            usage = "<String> Adds pre-processing step to list of steps."
    )
    public String addPreProcessingStep(CommandInterpreter ci, String step) {
        preProcessingSteps.add(step);
        return("Added pre-processing step to list of steps.");
    }

    @Command(
            usage = "<int> Remove pro-processing step at specified index (0-indexed)."
    )
    public String removePreProcessingStep(CommandInterpreter ci, int index) {
        preProcessingSteps.remove(index);
        return("Removed pre-processing step at specified index.");
    }

    @Command(
            usage = "Displays all added pre-processing steps."
    )
    public String viewPreProcessingSteps(CommandInterpreter ci) {
        for (int i = 0; i < preProcessingSteps.size(); i++) {
            System.out.println("\t" + i + ") "+ preProcessingSteps.get(i));
        }
        return("Displayed all added pre-processing steps.");
    }

    @Command(
            usage = "<String> Adds consideration to list of considerations."
    )
    public String addConsideration(CommandInterpreter ci, String consideration) {
        considerations.add(consideration);
        return("Added consideration to list of considerations.");
    }

    @Command(
            usage = "<int> Remove consideration at specified index (0-indexed)."
    )
    public String removeConsideration(CommandInterpreter ci, int index) {
        considerations.remove(index);
        return("Removed consideration at specified index.");
    }

    @Command(
            usage = "Displays all added considerations."
    )
    public String viewConsiderations(CommandInterpreter ci) {
        for (int i = 0; i < considerations.size(); i++) {
            System.out.println("\t" + i + ") "+ considerations.get(i));
        }
        return("Displayed all added considerations.");
    }

    @Command(
            usage = "<String> Adds relevant factor to list of factors."
    )
    public String addFactor(CommandInterpreter ci, String factor) {
        factors.add(factor);
        return("Added factor to list of factors.");
    }

    @Command(
            usage = "<int> Remove factor at specified index (0-indexed)."
    )
    public String removeFactor(CommandInterpreter ci, int index) {
        factors.remove(index);
        return("Removed factor at specified index.");
    }

    @Command(
            usage = "Displays all added factors."
    )
    public String viewFactors(CommandInterpreter ci) {
        for (int i = 0; i < factors.size(); i++) {
            System.out.println("\t" + i + ") "+ factors.get(i));
        }
        return("Displayed all added factors.");
    }

    @Command(
            usage = "<String> Adds resource to list of resources."
    )
    public String addResource(CommandInterpreter ci, String resource) {
        resources.add(resource);
        return("Added resource to list of resources.");
    }

    @Command(
            usage = "<int> Remove resource at specified index (0-indexed)."
    )
    public String removeResource(CommandInterpreter ci, int index) {
        resources.remove(index);
        return("Removed resource at specified index.");
    }

    @Command(
            usage = "Displays all added resources."
    )
    public String viewResources(CommandInterpreter ci) {
        for (int i = 0; i < resources.size(); i++) {
            System.out.println("\t" + i + ") "+ resources.get(i));
        }
        return("Displayed all added resources.");
    }

    @Command(
            usage = "<String> Records primary contact in case of questions or comments."
    )
    public String primaryContact(CommandInterpreter ci, String contact) {
        builder.primaryContact(contact);
        return("Recorded primary contact as " + contact + ".");
    }

    @Command(
            usage = "<String> Records model's citation."
    )
    public String modelCitation(CommandInterpreter ci, String citation) {
        builder.modelCitation(citation);
        return("Recorded model citation as " + citation + ".");
    }

    @Command(
            usage = "<String> Records model's license."
    )
    public String modelLicense(CommandInterpreter ci, String license) {
        builder.modelLicense(license);
        return("Recorded model license as " + license + ".");
    }

    private UsageDetails createUsageDetails() {
        builder.outOfScopeUses(outOfScopeUses);
        builder.preProcessingSteps(preProcessingSteps);
        builder.considerations(considerations);
        builder.factors(factors);
        builder.resources(resources);
        return builder.build();
    }

    @Command(
            usage = "<filename> Saves UsageDetails to an existing ModelCard file."
    )
    public String saveUsageDetails(CommandInterpreter ci, File destinationFile) throws IOException {
        UsageDetails usageDetails = createUsageDetails();

        ObjectNode usageDetailsObject = usageDetails.toJson();
        ObjectNode modelCardObject = mapper.readValue(destinationFile, ObjectNode.class);
        if (!modelCardObject.get("UsageDetails").isNull()) {
            throw new IllegalArgumentException("This ModelCard already contains a UsageDetails.");
        }
        modelCardObject.set("UsageDetails", usageDetailsObject);
        mapper.writeValue(destinationFile, modelCardObject);

        return "Saved UsageDetails to destination file.";
    }

    @Command(
            usage = "Removes all previously written fields for UsageDetails to write a new UsageDetails."
    )
    public String newUsageDetails(CommandInterpreter ci) {
        builder = new UsageDetailsBuilder();
        outOfScopeUses.clear();
        preProcessingSteps.clear();
        considerations.clear();
        factors.clear();
        resources.clear();
        return "Started a new UsageDetails.";
    }

    @Command(
            usage = "Displays current state of UsageDetails."
    )
    public String viewUsageDetails(CommandInterpreter ci) {
        System.out.println(createUsageDetails());
        return "Displayed current state of UsageDetails.";
    }


    @Command(
            usage = "Closes CLI without explicitly saving anything recorded."
    )
    public String close(CommandInterpreter ci) {
        shell.close();
        return "Closed ClI.";
    }

    public static void main(String[] args) {
        ModelCardCLI driver = new ModelCardCLI();
        driver.startShell();
    }
}
