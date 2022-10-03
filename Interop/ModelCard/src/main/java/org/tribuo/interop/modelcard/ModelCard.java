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
import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * ModelCard feature to allow more transparent model reporting.
 * <p>
 * See:
 * <pre>
 * M. Mitchell et al.
 * "Model Cards for Model Reporting"
 * In Conference in Fairness, Accountability, and Transparency, 2019.
 * </pre>
 * <p>
 * At the moment, the ModelCard system only supports models trained within Tribuo and throws an error for
 * all external models.
 */
public class ModelCard {
    static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final ModelDetails modelDetails;
    private final TrainingDetails trainingDetails;
    private final TestingDetails testingDetails;
    private final UsageDetails usageDetails;

    /**
     * Creates an instance of ModelCard.
     * <p>
     * Throws {@link IllegalArgumentException} if the model is an external model trained outside Tribuo.
     * @param model The trained model for which a model card will be built.
     * @param evaluation An {@link Evaluation} object for the trained model.
     * @param testingMetrics A map of metric descriptions paired with their corresponding metric values for the trained model.
     * @param usage A {@link UsageDetails} object specifying the usage details of the trained model.
     */
    public ModelCard(Model<?> model, Evaluation<?> evaluation, Map<String, Double> testingMetrics, UsageDetails usage) {
        if (model instanceof ExternalModel) {
            throw new IllegalArgumentException("External models currently not supported by ModelCard.");
        }
        modelDetails = new ModelDetails(model);
        trainingDetails = new TrainingDetails(model);
        testingDetails = new TestingDetails(evaluation, testingMetrics);
        usageDetails = usage;
    }

    /**
     * Creates an instance of ModelCard that does not include any extracted metrics.
     *
     * @param model The trained model for which a model card will be built.
     * @param evaluation An {@link Evaluation} object for the trained model.
     * @param usage A {@link UsageDetails} object specifying the usage details of the trained model.
     */
    public ModelCard(Model<?> model, Evaluation<?> evaluation, UsageDetails usage) {
        this(model, evaluation, Collections.emptyMap(), usage);
    }

    /**
     * Creates an instance of ModelCard that has its {@link UsageDetails} set to null.
     *
     * @param model The trained model for which a model card will be built.
     * @param evaluation An {@link Evaluation} object for the trained model.
     * @param testingMetrics A map of metric descriptions paired with their corresponding metric values for the trained model.
     */
    public ModelCard(Model<?> model, Evaluation<?> evaluation, Map<String, Double> testingMetrics) {
        this(model, evaluation, testingMetrics, null);
    }

    /**
     * Creates an instance of ModelCard that does not include any extracted metrics and has its {@link UsageDetails} set to null.
     *
     * @param model The trained model for which a model card will be built.
     * @param evaluation An {@link Evaluation} object for the trained model.
     */
    public ModelCard(Model<?> model, Evaluation<?> evaluation) {
        this(model, evaluation, Collections.emptyMap(), null);
    }

    /**
     * Creates an instance of ModelCard that does not include any testing metrics and has its {@link UsageDetails} set to null.
     *
     * @param model The trained model for which a model card will be built.
     */
    ModelCard(Model<?> model) {
        if (model instanceof ExternalModel) {
            throw new IllegalArgumentException("External models currently not supported by ModelCard.");
        }
        modelDetails = new ModelDetails(model);
        trainingDetails = new TrainingDetails(model);
        testingDetails = new TestingDetails();
        usageDetails = null;
    }

    /**
     * Creates an instance of ModelCard.
     * @param modelCard The Json content corresponding to a serialized ModelCard that will be used to recreate
     * a new instance of a ModelCard.
     * @throws JsonProcessingException if a problem is encountered when processing Json content.
     */
    private ModelCard(JsonNode modelCard) throws JsonProcessingException {
        modelDetails = new ModelDetails(modelCard.get("ModelDetails"));
        trainingDetails = new TrainingDetails(modelCard.get("TrainingDetails"));
        testingDetails = new TestingDetails(modelCard.get("TestingDetails"));
        if (modelCard.get("UsageDetails").isNull()) {
            usageDetails = null;
        } else {
            usageDetails = new UsageDetails(modelCard.get("UsageDetails"));
        }
    }

    /**
     * Reads the Json content corresponding to a ModelCard from file and instantiates it.
     * @param sourceFile The Json file path corresponding to a serialized ModelCard.
     * @return A {@link ModelCard} object corresponding to the provided serialized ModelCard.
     * @throws IOException If the model card could not be read from the path, or the Json failed to parse.
     */
    public static ModelCard deserializeFromJson(Path sourceFile) throws IOException {
        JsonNode modelCard = mapper.readTree(sourceFile.toFile());
        return new ModelCard(modelCard);
    }

    /**
     * Reads the Json content corresponding to a ModelCard and instantiates it.
     *
     * @param modelCard The Json content corresponding to a serialized ModelCard that will be used to recreate
     * a new instance of a ModelCard.
     * @return A ModelCard object corresponding to the provided serialized ModelCard.
     * @throws JsonProcessingException if a problem is encountered when processing Json content.
     */
    public static ModelCard deserializeFromJson(JsonNode modelCard) throws JsonProcessingException {
        return new ModelCard(modelCard);
    }

    /**
     * Gets the {@link ModelDetails} of the ModelCard object.
     * @return The {@link ModelDetails} of the ModelCard object.
     */
    public ModelDetails getModelDetails() {
        return modelDetails;
    }

    /**
     * Gets the {@link TrainingDetails} of the ModelCard object.
     * @return The {@link TrainingDetails} of the ModelCard object.
     */
    public TrainingDetails getTrainingDetails() {
        return trainingDetails;
    }

    /**
     * Gets the {@link TestingDetails} of the ModelCard object.
     * @return The {@link TestingDetails} of the ModelCard object.
     */
    public TestingDetails getTestingDetails() {
        return testingDetails;
    }

    /**
     * Gets the {@link UsageDetails} of the ModelCard object, which may be null.
     * @return The {@link UsageDetails} of the ModelCard object.
     */
    public UsageDetails getUsageDetails() {
        return usageDetails;
    }

    /**
     * Creates a Json object corresponding this ModelCard instance.
     * @return The {@link ObjectNode} corresponding to this ModelCard instance.
     */
    public ObjectNode toJson() {
        ObjectNode modelCardObject = mapper.createObjectNode();
        modelCardObject.set("ModelDetails", modelDetails.toJson());
        modelCardObject.set("TrainingDetails", trainingDetails.toJson());
        modelCardObject.set("TestingDetails", testingDetails.toJson());
        if (usageDetails != null) {
            modelCardObject.set("UsageDetails", usageDetails.toJson());
        } else {
            modelCardObject.putNull("UsageDetails");
        }
        return modelCardObject;
    }

    /**
     * Serializes and saves the ModelCard object to the specified path.
     *
     * @param destinationFile The file path to which the serialized ModelCard will be saved.
     * @throws IOException if a problem is encountered when processing Json content or writing the file.
     */
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
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ModelCard modelCard = (ModelCard) o;
        return modelDetails.equals(modelCard.modelDetails) &&
                trainingDetails.equals(modelCard.trainingDetails) &&
                testingDetails.equals(modelCard.testingDetails) &&
                Objects.equals(usageDetails, modelCard.usageDetails);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelDetails, trainingDetails, testingDetails, usageDetails);
    }
}
