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
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.Model;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

/**
 * ModelDetails section of a {@link ModelCard}.
 */
public final class ModelDetails {
    private static final String schemaVersion = "1.0";
    private final String modelType;
    private final String modelPackage;
    private final String tribuoVersion;
    private final String javaVersion;
    private final Map<String, Object> configuredParams;

    /**
     * Creates an instance of ModelDetails.
     * @param model The trained {@link Model} for which a ModelDetails will be built.
     */
    public ModelDetails(Model<?> model) {
        modelType = model.getClass().getSimpleName();
        modelPackage = model.getClass().getTypeName();
        tribuoVersion = model.getProvenance().getTribuoVersion();
        javaVersion = model.getProvenance().getJavaVersion();
        configuredParams = ProvenanceUtil.convertToMap(model.getProvenance().getTrainerProvenance());
    }

    /**
     * Creates an instance of ModelDetails.
     *
     * @param modelDetailsJson The Json content corresponding to a serialized ModelDetails that will be used to recreate
     * a new instance of a ModelDetails.
     * @throws JsonProcessingException if a problem is encountered when processing Json content.
     */
    public ModelDetails(JsonNode modelDetailsJson) throws JsonProcessingException {
        modelType = modelDetailsJson.get("model-type").textValue();
        modelPackage = modelDetailsJson.get("model-package").textValue();
        tribuoVersion = modelDetailsJson.get("tribuo-version").textValue();
        javaVersion = modelDetailsJson.get("java-version").textValue();
        TypeReference<Map<String, Object>> typeRef = new TypeReference<>() {};
        configuredParams = Collections.unmodifiableMap(mapper.readValue(modelDetailsJson.get("configured-parameters").toString(), typeRef));
    }

    /**
     * Gets the schema version of the ModelDetails object.
     * @return A string specifying the schema version of the ModelDetails object.
     */
    public String getSchemaVersion() {
        return schemaVersion;
    }

    /**
     * Gets the model type of the ModelDetails object.
     * @return A string specifying the model type of the ModelDetails object.
     */
    public String getModelType() {
        return modelType;
    }

    /**
     * Gets the model package of the ModelDetails object.
     * @return A string specifying the model package of the ModelDetails object.
     */
    public String getModelPackage() {
        return modelPackage;
    }

    /**
     * Gets the Tribuo version of the ModelDetails object.
     * @return A string specifying the Tribuo version of the ModelDetails object.
     */
    public String getTribuoVersion() {
        return tribuoVersion;
    }

    /**
     * Gets the Java version of the ModelDetails object.
     * @return A string specifying the Java version of the ModelDetails object.
     */
    public String getJavaVersion() {
        return javaVersion;
    }

    /**
     * Gets the configured parameters of the ModelDetails object.
     * @return An unmodifiable map of the configured parameters of the ModelDetails object.
     */
    public Map<String, Object> getConfiguredParams() {
        return Collections.unmodifiableMap(configuredParams);
    }

    /**
     * Creates a Json object corresponding this ModelDetails instance.
     * @return The {@link ObjectNode} corresponding to this ModelDetails instance.
     */
    public ObjectNode toJson() {
        ObjectNode modelDetailsObject = mapper.createObjectNode();
        modelDetailsObject.put("schema-version", schemaVersion);
        modelDetailsObject.put("model-type", modelType);
        modelDetailsObject.put("model-package", modelPackage);
        modelDetailsObject.put("tribuo-version", tribuoVersion);
        modelDetailsObject.put("java-version", javaVersion);
        modelDetailsObject.set("configured-parameters", mapper.convertValue(configuredParams, ObjectNode.class));
        return modelDetailsObject;
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
        ModelDetails that = (ModelDetails) o;
        return modelType.equals(that.modelType) &&
                modelPackage.equals(that.modelPackage) &&
                tribuoVersion.equals(that.tribuoVersion) &&
                javaVersion.equals(that.javaVersion) &&
                configuredParams.equals(that.configuredParams);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelType, modelPackage, tribuoVersion, javaVersion, configuredParams);
    }
}
