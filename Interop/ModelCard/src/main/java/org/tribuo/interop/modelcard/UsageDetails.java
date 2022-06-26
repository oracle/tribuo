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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

public final class UsageDetails implements CommandGroup {
    private final CommandInterpreter shell = new CommandInterpreter();
    public static final String schemaVersion = "1.0";
    private String intendedUse;
    private String intendedUsers;
    private final List<String> outOfScopeUses = new ArrayList<>();
    private final List<String> preProcessingSteps = new ArrayList<>();
    private final List<String> considerations = new ArrayList<>();
    private final List<String> factors = new ArrayList<>();
    private final List<String> resources = new ArrayList<>();
    private String primaryContact;
    private String modelCitation;
    private String modelLicense;

    public UsageDetails() {
        intendedUse = "";
        intendedUsers = "";
        primaryContact = "";
        modelCitation = "";
        modelLicense = "";
        shell.setPrompt("CLI% ");
    }

    public UsageDetails(JsonNode usageDetailsJson) {
        intendedUse = usageDetailsJson.get("intended-use").textValue();
        intendedUsers = usageDetailsJson.get("intended-users").textValue();

        for (int i = 0; i < usageDetailsJson.get("out-of-scope-uses").size(); i++) {
            outOfScopeUses.add(usageDetailsJson.get("out-of-scope-uses").get(i).textValue());
        }
        for (int i = 0; i < usageDetailsJson.get("pre-processing-steps").size(); i++) {
            preProcessingSteps.add(usageDetailsJson.get("pre-processing-steps").get(i).textValue());
        }
        for (int i = 0; i < usageDetailsJson.get("considerations-list").size(); i++) {
            considerations.add(usageDetailsJson.get("considerations-list").get(i).textValue());
        }
        for (int i = 0; i < usageDetailsJson.get("relevant-factors-list").size(); i++) {
            factors.add(usageDetailsJson.get("relevant-factors-list").get(i).textValue());
        }
        for (int i = 0; i < usageDetailsJson.get("resources-list").size(); i++) {
            resources.add(usageDetailsJson.get("resources-list").get(i).textValue());
        }
        primaryContact = usageDetailsJson.get("primary-contact").textValue();
        modelCitation = usageDetailsJson.get("model-citation").textValue();
        modelLicense = usageDetailsJson.get("model-license").textValue();
    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public String getIntendedUse() {
        return intendedUse;
    }

    public String getIntendedUsers() {
        return intendedUsers;
    }

    public List<String> getOutOfScopeUses() {
        return Collections.unmodifiableList(outOfScopeUses);
    }

    public List<String> getPreProcessingSteps() {
        return Collections.unmodifiableList(preProcessingSteps);
    }

    public List<String> getConsiderations() {
        return Collections.unmodifiableList(considerations);
    }

    public List<String> getFactors() {
        return Collections.unmodifiableList(factors);
    }

    public List<String> getResources() {
        return Collections.unmodifiableList(resources);
    }

    public String getPrimaryContact() {
        return primaryContact;
    }

    public String getModelCitation() {
        return modelCitation;
    }

    public String getModelLicense() {
        return modelLicense;
    }

    public void startShell() {
        shell.add(this);
        shell.start();
    }

    public ObjectNode toJson() {
        ObjectNode usageDetailsObject = mapper.createObjectNode();
        usageDetailsObject.put("schema-version", schemaVersion);
        usageDetailsObject.put("intended-use", intendedUse);
        usageDetailsObject.put("intended-users", intendedUsers);

        ArrayNode usesArr = mapper.createArrayNode();
        for (String s : outOfScopeUses) {
            usesArr.add(s);
        }
        usageDetailsObject.set("out-of-scope-uses", usesArr);

        ArrayNode processingArr = mapper.createArrayNode();
        for (String s : preProcessingSteps) {
            processingArr.add(s);
        }
        usageDetailsObject.set("pre-processing-steps", processingArr);

        ArrayNode considerationsArr = mapper.createArrayNode();
        for (String s : considerations) {
            considerationsArr.add(s);
        }
        usageDetailsObject.set("considerations-list", considerationsArr);

        ArrayNode factorsArr = mapper.createArrayNode();
        for (String s : factors) {
            factorsArr.add(s);
        }
        usageDetailsObject.set("relevant-factors-list", factorsArr);

        ArrayNode resourcesArr = mapper.createArrayNode();
        for (String s : resources) {
            resourcesArr.add(s);
        }
        usageDetailsObject.set("resources-list", resourcesArr);

        usageDetailsObject.put("primary-contact", primaryContact);
        usageDetailsObject.put("model-citation", modelCitation);
        usageDetailsObject.put("model-license", modelLicense);

        return usageDetailsObject;
    }

    @Override
    public String getName() {
        return "UsageDetails";
    }

    @Override
    public String getDescription() {
        return "Commands for specifying UsageDetails for a model.";
    }

    @Command(
            usage = "<String> Records intended use of model."
    )
    public String intendedUse(CommandInterpreter ci, String use) {
        intendedUse = use;
        return("Recorded intended use as " + intendedUse + ".");
    }

    @Command(
            usage = "<String> Records intended users of model."
    )
    public String intendedUsers(CommandInterpreter ci, String users) {
        intendedUsers = users;
        return("Recorded intended users as " + intendedUsers + ".");
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
        primaryContact = contact;
        return("Recorded primary contact as " + primaryContact + ".");
    }

    @Command(
            usage = "<String> Records model's citation."
    )
    public String modelCitation(CommandInterpreter ci, String citation) {
        modelCitation = citation;
        return("Recorded model citation as " + modelCitation + ".");
    }

    @Command(
            usage = "<String> Records model's license."
    )
    public String modelLicense(CommandInterpreter ci, String license) {
        modelLicense = license;
        return("Recorded model license as " + modelLicense + ".");
    }

    @Command(
            usage = "<filename> Saves UsageDetails to an existing ModelCard file."
    )
    public String saveUsageDetails(CommandInterpreter ci, File destinationFile) throws IOException {
        ObjectNode modelCardObject = mapper.readValue(destinationFile, ObjectNode.class);
        ObjectNode usageDetailsObject = toJson();
        modelCardObject.set("UsageDetails", usageDetailsObject);
        mapper.writeValue(destinationFile, modelCardObject);
        return "Saved UsageDetails to destination file.";
    }

    @Command(
            usage = "Closes shell without saving any recorded content."
    )
    public String close(CommandInterpreter ci) {
        shell.close();
        return "Closed shell.";
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        UsageDetails that = (UsageDetails) o;
        return intendedUse.equals(that.intendedUse) &&
                intendedUsers.equals(that.intendedUsers) &&
                outOfScopeUses.equals(that.outOfScopeUses) &&
                preProcessingSteps.equals(that.preProcessingSteps) &&
                considerations.equals(that.considerations) &&
                factors.equals(that.factors) &&
                resources.equals(that.resources) &&
                primaryContact.equals(that.primaryContact) &&
                modelCitation.equals(that.modelCitation) &&
                modelLicense.equals(that.modelLicense);
    }

    @Override
    public int hashCode() {
        return Objects.hash(intendedUse, intendedUsers, outOfScopeUses, preProcessingSteps, considerations, factors, resources, primaryContact, modelCitation, modelLicense);
    }

    public static void main(String[] args) {
        UsageDetails driver = new UsageDetails();
        driver.startShell();
    }
}