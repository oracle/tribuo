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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

public final class UsageDetails {
    public static final String schemaVersion = "1.0";
    private final String intendedUse;
    private final String intendedUsers;
    private final List<String> outOfScopeUses;
    private final List<String> preProcessingSteps;
    private final List<String> considerations;
    private final List<String> factors;
    private final List<String> resources;
    private final String primaryContact;
    private final String modelCitation;
    private final String modelLicense;

    public UsageDetails(
            String intendedUse,
            String intendedUsers,
            List<String> outOfScopeUses,
            List<String> preProcessingSteps,
            List<String> considerations,
            List<String> factors,
            List<String> resources,
            String primaryContact,
            String modelCitation,
            String modelLicense)
    {
        this.intendedUse = intendedUse;
        this.intendedUsers = intendedUsers;
        this.outOfScopeUses = outOfScopeUses;
        this.preProcessingSteps = preProcessingSteps;
        this.considerations = considerations;
        this.factors = factors;
        this.resources = resources;
        this.primaryContact = primaryContact;
        this.modelCitation = modelCitation;
        this.modelLicense = modelLicense;
    }

    public UsageDetails(JsonNode usageDetailsJson) {
        intendedUse = usageDetailsJson.get("intended-use").textValue();
        intendedUsers = usageDetailsJson.get("intended-users").textValue();

        outOfScopeUses = new ArrayList<>();
        for (int i = 0; i < usageDetailsJson.get("out-of-scope-uses").size(); i++) {
            outOfScopeUses.add(usageDetailsJson.get("out-of-scope-uses").get(i).textValue());
        }
        preProcessingSteps = new ArrayList<>();
        for (int i = 0; i < usageDetailsJson.get("pre-processing-steps").size(); i++) {
            preProcessingSteps.add(usageDetailsJson.get("pre-processing-steps").get(i).textValue());
        }
        considerations = new ArrayList<>();
        for (int i = 0; i < usageDetailsJson.get("considerations-list").size(); i++) {
            considerations.add(usageDetailsJson.get("considerations-list").get(i).textValue());
        }
        factors = new ArrayList<>();
        for (int i = 0; i < usageDetailsJson.get("relevant-factors-list").size(); i++) {
            factors.add(usageDetailsJson.get("relevant-factors-list").get(i).textValue());
        }
        resources = new ArrayList<>();
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
    public String toString() {
        return toJson().toPrettyString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
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
}