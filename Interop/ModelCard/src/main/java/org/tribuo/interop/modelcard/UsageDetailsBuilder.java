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

/**
 * A builder class for creating an instance of {@link UsageDetails}.
 */
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

    /**
     * Creates an instance of UsageDetails.
     */
    public UsageDetailsBuilder() { }

    /**
     * Gets the intended use of the model for which an instance of UsageDetails will be built.
     * @return A string specifying the intended use of the model.
     */
    public String getIntendedUse() {
        return intendedUse;
    }

    /**
     * Gets the intended users of the model for which an instance of UsageDetails will be built.
     * @return A string specifying the intended users of the model.
     */
    public String getIntendedUsers() {
        return intendedUsers;
    }

    /**
     * Gets the out-of-scope uses of the model for which an instance of UsageDetails will be built.
     * @return A list of out-of-scope uses for the model.
     */
    public List<String> getOutOfScopeUses() {
        return Collections.unmodifiableList(outOfScopeUses);
    }

    /**
     * Gets the pre-processing steps of the model for which an instance of UsageDetails will be built.
     * @return A list of pre-processing steps for the model.
     */
    public List<String> getPreProcessingSteps() {
        return Collections.unmodifiableList(preProcessingSteps);
    }

    /**
     * Gets the considerations of the model for which an instance of UsageDetails will be built.
     * @return A list of considerations for the model.
     */
    public List<String> getConsiderations() {
        return Collections.unmodifiableList(considerations);
    }

    /**
     * Gets the relevant factors of the model for which an instance of UsageDetails will be built.
     * @return A list of relevant factors for the model.
     */
    public List<String> getFactors() {
        return Collections.unmodifiableList(factors);
    }

    /**
     * Gets the relevant resources of the model for which an instance of UsageDetails will be built.
     * @return A list of relevant resources for the model.
     */
    public List<String> getResources() {
        return Collections.unmodifiableList(resources);
    }

    /**
     * Gets the primary contact person of the model for which an instance of UsageDetails will be built.
     * @return A string specifying the primary contact person of the model.
     */
    public String getPrimaryContact() {
        return primaryContact;
    }

    /**
     * Gets the model citation of the model for which an instance of UsageDetails will be built.
     * @return A string specifying the model citation of the model.
     */
    public String getModelCitation() {
        return modelCitation;
    }

    /**
     * Gets the model license of the model for which an instance of UsageDetails will be built.
     * @return A string specifying the model license of the model.
     */
    public String getModelLicense() {
        return modelLicense;
    }

    /**
     * Sets the intended use of the model for which an instance of UsageDetails will be built.
     * @param intendedUse The intented use of the model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder intendedUse(String intendedUse) {
        this.intendedUse = intendedUse;
        return this;
    }

    /**
     * Sets the intended users of the model for which an instance of UsageDetails will be built.
     * @param intendedUsers The intended users of the model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder intendedUsers(String intendedUsers) {
        this.intendedUsers = intendedUsers;
        return this;
    }

    /**
     * Sets the out-of-scope uses of the model for which an instance of UsageDetails will be built.
     * @param uses Out of scope uses of this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder outOfScopeUses(List<String> uses) {
        this.outOfScopeUses.addAll(uses);
        return this;
    }

    /**
     * Sets the pre-processing steps of the model for which an instance of UsageDetails will be built.
     * @param steps Pre-processing steps applied to the data before the model was built.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder preProcessingSteps(List<String> steps) {
        this.preProcessingSteps.addAll(steps);
        return this;
    }

    /**
     * Sets the considerations of the model for which an instance of UsageDetails will be built.
     * @param considerations Considerations for using this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder considerations(List<String> considerations) {
        this.considerations.addAll(considerations);
        return this;
    }

    /**
     * Sets the relevant factors of the model for which an instance of UsageDetails will be built.
     * @param factors Relevant factors when considering this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder factors(List<String> factors) {
        this.factors.addAll(factors);
        return this;
    }

    /**
     * Sets the relevant resources of the model for which an instance of UsageDetails will be built.
     * @param resources Relevant resources when using this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder resources(List<String> resources) {
        this.resources.addAll(resources);
        return this;
    }

    /**
     * Sets the primary contact person of the model for which an instance of UsageDetails will be built.
     * @param primaryContact The primary contact for this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder primaryContact(String primaryContact) {
        this.primaryContact = primaryContact;
        return this;
    }

    /**
     * Sets the model citation of the model for which an instance of UsageDetails will be built.
     * @param modelCitation A citation which can be used for this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder modelCitation(String modelCitation) {
        this.modelCitation = modelCitation;
        return this;
    }

    /**
     * Sets the model license of the model for which an instance of UsageDetails will be built.
     * @param modelLicense The license for this model.
     * @return This instance of UsageDetailsBuilder.
     */
    public UsageDetailsBuilder modelLicense(String modelLicense) {
        this.modelLicense = modelLicense;
        return this;
    }

    /**
     * Builds an instance of {@link UsageDetails} using the recorded field values or their default values.
     * @return An instance of {@link UsageDetails}.
     */
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
