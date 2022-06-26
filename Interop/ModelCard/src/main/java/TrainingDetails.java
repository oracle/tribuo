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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.Model;
import org.tribuo.regression.Regressor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class TrainingDetails {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);;
    public static final String schemaVersion = "1.0";
    private final String trainingTime;
    private final int trainingSetSize;
    private final int numFeatures;
    private final List<String> features = new ArrayList<>();;
    private final int numOutputs;
    private final Map<String, Long> outputsDistribution = new HashMap<>();

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

    public TrainingDetails(JsonNode trainingDetailsJson) throws JsonProcessingException {
        trainingTime = trainingDetailsJson.get("training-time").textValue();
        trainingSetSize = trainingDetailsJson.get("training-set-size").intValue();

        numFeatures = trainingDetailsJson.get("num-features").intValue();
        for (int i = 0; i < trainingDetailsJson.get("features-list").size(); i++) {
            features.add(trainingDetailsJson.get("features-list").get(i).textValue());
        }

        numOutputs = trainingDetailsJson.get("num-outputs").intValue();
        Map<String, Integer> parsed = mapper.readValue(trainingDetailsJson.get("outputs-distribution").toString(), Map.class);
        for (var entry : parsed.keySet()) {
            outputsDistribution.put(entry, parsed.get(entry).longValue());
        }
    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public String getTrainingTime() {
        return trainingTime;
    }

    public int getTrainingSetSize() {
        return trainingSetSize;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public List<String> getFeatures() {
        return features;
    }

    public int getNumOutputs() {
        return numOutputs;
    }

    public Map<String, Long> getOutputsDistribution() {
        return outputsDistribution;
    }

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
        for (String description : outputsDistribution.keySet()) {
            outputsArr.put(description, outputsDistribution.get(description));
        }
        datasetDetailsObject.set("outputs-distribution", outputsArr);

        return datasetDetailsObject;
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }
}