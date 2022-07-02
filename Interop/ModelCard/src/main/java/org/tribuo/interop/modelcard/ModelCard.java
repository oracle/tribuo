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
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.Model;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.interop.ExternalModel;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

public class ModelCard {
    static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final ModelDetails modelDetails;
    private final TrainingDetails trainingDetails;
    private final TestingDetails testingDetails;
    private final UsageDetails usageDetails;

    public ModelCard(Model<?> model, Evaluation<?> evaluation) {
        if (model instanceof ExternalModel) {
            throw new IllegalArgumentException("External models currently not supported by ModelCard.");
        }
        modelDetails = new ModelDetails(model);
        trainingDetails = new TrainingDetails(model);
        testingDetails = new TestingDetails(evaluation);
        usageDetails = null;
    }

    public ModelCard(Model<?> model, Evaluation<?> evaluation, UsageDetails usage) {
        if (model instanceof ExternalModel) {
            throw new IllegalArgumentException("External models currently not supported by ModelCard.");
        }
        modelDetails = new ModelDetails(model);
        trainingDetails = new TrainingDetails(model);
        testingDetails = new TestingDetails(evaluation);
        usageDetails = usage;
    }

    private ModelCard(JsonNode modelCard) throws JsonProcessingException {
        modelDetails = new ModelDetails(modelCard.get("ModelDetails"));
        trainingDetails = new TrainingDetails(modelCard.get("TrainingDetails"));
        testingDetails = new TestingDetails(modelCard.get("TestingDetails"));
        usageDetails = new UsageDetails(modelCard.get("UsageDetails"));
    }

    public static ModelCard deserializeFromJson(Path sourceFile) throws IOException {
        JsonNode modelCard = mapper.readTree(sourceFile.toFile());
        return new ModelCard(modelCard);
    }

    public static ModelCard deserializeFromJson(JsonNode modelCard) throws JsonProcessingException {
        return new ModelCard(modelCard);
    }

    public ModelDetails getModelDetails() {
        return modelDetails;
    }

    public TrainingDetails getTrainingDetails() {
        return trainingDetails;
    }

    public TestingDetails getTestingDetails() {
        return testingDetails;
    }

    public UsageDetails getUsageDetails() {
        return usageDetails;
    }

    public void addMetric(String metricDescription, Double metricValue) {
        testingDetails.addMetric(metricDescription, metricValue);
    }

    public ObjectNode toJson() {
        ObjectNode modelCardObject = mapper.createObjectNode();
        modelCardObject.set("ModelDetails", modelDetails.toJson());
        modelCardObject.set("TrainingDetails", trainingDetails.toJson());
        modelCardObject.set("TestingDetails", testingDetails.toJson());
        if (usageDetails != null) {
            modelCardObject.set("UsageDetails", usageDetails.toJson());
        }
        return modelCardObject;
    }

    public void saveToFile(Path destinationFile) throws IOException {
        ObjectNode modelCardObject = toJson();
        mapper.writeValue(destinationFile.toFile(), modelCardObject);
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
        ModelCard modelCard = (ModelCard) o;
        return modelDetails.equals(modelCard.modelDetails) &&
                trainingDetails.equals(modelCard.trainingDetails) &&
                testingDetails.equals(modelCard.testingDetails) &&
                usageDetails.equals(modelCard.usageDetails);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelDetails, trainingDetails, testingDetails, usageDetails);
    }
}
