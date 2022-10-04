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
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.Model;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

/**
 * TrainingDetails section of a {@link ModelCard}.
 */
public final class TrainingDetails {
    private static final String schemaVersion = "1.0";
    private final String trainingTime;
    private final int trainingSetSize;
    private final int numFeatures;
    private final List<String> features = new ArrayList<>();;
    private final int numOutputs;
    private final Map<String, Long> outputsDistribution = new HashMap<>();

    /**
     * Creates an instance of TrainingDetails.
     * @param model The trained {@link Model} for which a TrainingDetails will be built.
     */
    public TrainingDetails(Model<?> model) {
        trainingTime = model.getProvenance().getTrainingTime().toString();
        trainingSetSize = model.getProvenance().getDatasetProvenance().getNumExamples();

        numFeatures = model.getProvenance().getDatasetProvenance().getNumFeatures();
        for (int i = 0; i < model.getFeatureIDMap().size(); i++) {
            features.add(model.getFeatureIDMap().get(i).getName());
        }

        numOutputs = model.getProvenance().getDatasetProvenance().getNumOutputs();

        if (!model.validate(Regressor.class)) {
            for (var pair : model.getOutputIDInfo().outputCountsIterable()) {
                outputsDistribution.put(pair.getA(), pair.getB());
            }
        }
    }

    /**
     * Creates an instance of TrainingDetails.
     *
     * @param trainingDetailsJson The Json content corresponding to a serialized TrainingDetails that will be used to
     * recreate a new instance of a TrainingDetails.
     * @throws JsonProcessingException if a problem is encountered when processing Json content.
     */
    public TrainingDetails(JsonNode trainingDetailsJson) throws JsonProcessingException {
        trainingTime = trainingDetailsJson.get("training-time").textValue();
        trainingSetSize = trainingDetailsJson.get("training-set-size").intValue();

        numFeatures = trainingDetailsJson.get("num-features").intValue();
        for (int i = 0; i < trainingDetailsJson.get("features-list").size(); i++) {
            features.add(trainingDetailsJson.get("features-list").get(i).textValue());
        }

        numOutputs = trainingDetailsJson.get("num-outputs").intValue();
        Map<?,?> parsed = mapper.readValue(trainingDetailsJson.get("outputs-distribution").toString(), Map.class);
        for (Map.Entry<?,?> entry : parsed.entrySet()) {
            Integer val = (Integer) entry.getValue();
            outputsDistribution.put((String)entry.getKey(), val.longValue());
        }
    }

    /**
     * Gets the schema version of the TrainingDetails object.
     * @return A string specifying the schema version of the TrainingDetails object.
     */
    public String getSchemaVersion() {
        return schemaVersion;
    }

    /**
     * Gets the training time of the TrainingDetails object.
     * @return A string specifying the training time of the TrainingDetails object.
     */
    public String getTrainingTime() {
        return trainingTime;
    }

    /**
     * Gets the training set size of the TrainingDetails object.
     * @return An int specifying the training set size of the TrainingDetails object.
     */
    public int getTrainingSetSize() {
        return trainingSetSize;
    }

    /**
     * Gets the number of features in the training set of the TrainingDetails object.
     * @return An int specifying the number of features of the TrainingDetails object.
     */
    public int getNumFeatures() {
        return numFeatures;
    }

    /**
     * Gets the list of features of the TrainingDetails object.
     * @return An unmodifiable list of the features of the TrainingDetails object.
     */
    public List<String> getFeatures() {
        return Collections.unmodifiableList(features);
    }

    /**
     * Gets the number of outputs in the training set of the TrainingDetails object.
     * @return An int specifying the number of outputs of the TrainingDetails object.
     */
    public int getNumOutputs() {
        return numOutputs;
    }

    /**
     * Gets the distribution of the outputs of the TrainingDetails object.
     * @return An unmodifiable map of the distribution of the outputs for the TrainingDetails object.
     */
    public Map<String, Long> getOutputsDistribution() {
        return Collections.unmodifiableMap(outputsDistribution);
    }

    /**
     * Creates a Json object corresponding this TrainingDetails instance.
     * @return The {@link ObjectNode} corresponding to this TrainingDetails instance.
     */
    public ObjectNode toJson() {
        ObjectNode datasetDetailsObject = mapper.createObjectNode();
        datasetDetailsObject.put("schema-version", schemaVersion);
        datasetDetailsObject.put("training-time", trainingTime);
        datasetDetailsObject.put("training-set-size", trainingSetSize);

        datasetDetailsObject.put("num-features", numFeatures);
        ArrayNode featuresArr = mapper.createArrayNode();
        for (String s : features) {
            featuresArr.add(s);
        }
        datasetDetailsObject.set("features-list", featuresArr);

        datasetDetailsObject.put("num-outputs", numOutputs);
        ObjectNode outputsArr = mapper.createObjectNode();
        for (Map.Entry<String, Long> entry : outputsDistribution.entrySet()) {
            outputsArr.put(entry.getKey(), entry.getValue());
        }
        datasetDetailsObject.set("outputs-distribution", outputsArr);

        return datasetDetailsObject;
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
        TrainingDetails that = (TrainingDetails) o;
        return trainingSetSize == that.trainingSetSize &&
                numFeatures == that.numFeatures &&
                numOutputs == that.numOutputs &&
                trainingTime.equals(that.trainingTime) &&
                features.equals(that.features) &&
                outputsDistribution.equals(that.outputsDistribution);
    }

    @Override
    public int hashCode() {
        return Objects.hash(trainingTime, trainingSetSize, numFeatures, features, numOutputs, outputsDistribution);
    }
}