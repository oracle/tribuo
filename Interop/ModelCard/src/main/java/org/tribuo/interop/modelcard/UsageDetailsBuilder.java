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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class UsageDetailsBuilder {
    private String intendedUse = "";
    private String intendedUsers = "";
    private final List<String> outOfScopeUses = new ArrayList<>();
    private final List<String> preProcessingSteps = new ArrayList<>();
    private final List<String> considerations = new ArrayList<>();
    private final List<String> factors = new ArrayList<>();
    private final List<String> resources = new ArrayList<>();
    private String primaryContact = "";
    private String modelCitation = "";
    private String modelLicense = "";

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

    public UsageDetailsBuilder intendedUse(String intendedUse) {
        this.intendedUse = intendedUse;
        return this;
    }

    public UsageDetailsBuilder intendedUsers(String intendedUsers) {
        this.intendedUsers = intendedUsers;
        return this;
    }

    public UsageDetailsBuilder outOfScopeUses(List<String> uses) {
        this.outOfScopeUses.addAll(uses);
        return this;
    }

    public UsageDetailsBuilder preProcessingSteps(List<String> steps) {
        this.preProcessingSteps.addAll(steps);
        return this;
    }

    public UsageDetailsBuilder considerations(List<String> considerations) {
        this.considerations.addAll(considerations);
        return this;
    }

    public UsageDetailsBuilder factors(List<String> factors) {
        this.factors.addAll(factors);
        return this;
    }

    public UsageDetailsBuilder resources(List<String> resources) {
        this.resources.addAll(resources);
        return this;
    }

    public UsageDetailsBuilder primaryContact(String primaryContact) {
        this.primaryContact = primaryContact;
        return this;
    }

    public UsageDetailsBuilder modelCitation(String modelCitation) {
        this.modelCitation = modelCitation;
        return this;
    }

    public UsageDetailsBuilder modelLicense(String modelLicense) {
        this.modelLicense = modelLicense;
        return this;
    }

    public UsageDetails build() {
        return new UsageDetails(
                intendedUse,
                intendedUsers,
                outOfScopeUses,
                preProcessingSteps,
                considerations,
                factors,
                resources,
                primaryContact,
                modelCitation,
                modelLicense
        );
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        UsageDetailsBuilder that = (UsageDetailsBuilder) o;
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
