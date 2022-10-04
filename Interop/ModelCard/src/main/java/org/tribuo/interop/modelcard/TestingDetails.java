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
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.evaluation.Evaluation;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static org.tribuo.interop.modelcard.ModelCard.mapper;

/**
 * TestingDetails section of a {@link ModelCard}.
 */
public final class TestingDetails {
    private static final String schemaVersion = "1.0";
    private final int testingSetSize;
    private final Map<String, Double> metrics = new HashMap<>();

    /**
     * Creates an instance of TestingDetails.
     * @param evaluation The {@link Evaluation} object for which a TestingDetails will be built.
     * @param testingMetrics The map of metric descriptions and values that will be recorded in the TestingDetails object.
     */
    public TestingDetails(Evaluation<?> evaluation, Map<String, Double> testingMetrics) {
        testingSetSize = evaluation.getPredictions().size();
        metrics.putAll(testingMetrics);
    }

    /**
     * Creates an instance of TestingDetails.
     * @param evaluation The {@link Evaluation} object for which a TestingDetails will be built.
     */
    public TestingDetails(Evaluation<?> evaluation) {
        this(evaluation, Collections.emptyMap());
    }

    /**
     * Creates an empty TestingDetails.
     */
    TestingDetails() {
        testingSetSize = 0;
    }

    /**
     * Creates an instance of TestingDetails.
     *
     * @param testingDetailsJson The Json content corresponding to a serialized TestingDetails that will be used to
     * recreate a new instance of a TestingDetails.
     * @throws JsonProcessingException if a problem is encountered when processing Json content.
     */
    public TestingDetails(JsonNode testingDetailsJson) throws JsonProcessingException {
        testingSetSize = testingDetailsJson.get("testing-set-size").intValue();
        Map<?, ?> parsed = mapper.readValue(testingDetailsJson.get("metrics").toString(), Map.class);
        for (Map.Entry<?,?> entry : parsed.entrySet()) {
            metrics.put((String) entry.getKey(), (Double) entry.getValue());
        }

    }

    /**
     * Gets the schema version of the TestingDetails object.
     * @return A string specifying the schema version of the TestingDetails object.
     */
    public String getSchemaVersion() {
        return schemaVersion;
    }

    /**
     * Gets the testing set size of the TestingDetails object.
     * @return An int specifying the testing set size of the TestingDetails object.
     */
    public int getTestingSetSize() {
        return testingSetSize;
    }

    /**
     * Gets the map of metric descriptions and values of the TestingDetails object.
     * @return An unmodifiable map of the metric descriptions and values of the TestingDetails object.
     */
    public Map<String, Double> getMetrics() {
        return Collections.unmodifiableMap(metrics);
    }

    /**
     * Creates a Json object corresponding this TestingDetails instance.
     * @return The {@link ObjectNode} corresponding to this TestingDetails instance.
     */
    public ObjectNode toJson() {
        ObjectNode testingDetailsObject = mapper.createObjectNode();
        testingDetailsObject.put("schema-version", schemaVersion);
        testingDetailsObject.put("testing-set-size", testingSetSize);

        ObjectNode testingMetricsObject = mapper.createObjectNode();
        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            testingMetricsObject.put(entry.getKey(), entry.getValue());
        }
        testingDetailsObject.set("metrics", testingMetricsObject);

        return testingDetailsObject;
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
        TestingDetails that = (TestingDetails) o;
        return testingSetSize == that.testingSetSize && metrics.equals(that.metrics);
    }

    @Override
    public int hashCode() {
        return Objects.hash(testingSetSize, metrics);
    }
}

