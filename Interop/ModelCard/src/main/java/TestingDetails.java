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
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.evaluation.Evaluation;

import java.util.HashMap;
import java.util.Map;

public final class TestingDetails {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private static final String schemaVersion = "1.0";
    private final int testingSetSize;
    private final Map<String, Double> metrics = new HashMap<>();

    public TestingDetails(Evaluation<?> evaluation) {
        testingSetSize = evaluation.getPredictions().size();
    }

    public TestingDetails(JsonNode testingDetailsJson) throws JsonProcessingException {
        testingSetSize = testingDetailsJson.get("testing-set-size").intValue();
        Map<String, Double> parsed = mapper.readValue(testingDetailsJson.get("metrics").toString(), Map.class);
        for (var entry : parsed.keySet()) {
            metrics.put(entry, parsed.get(entry));
        }

    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public int getTestingSetSize() {
        return testingSetSize;
    }

    public Map<String, Double> getMetrics() {
        return metrics;
    }

    public void addMetric(String metricDescription, Double metricValue) {
        metrics.put(metricDescription, metricValue);
    }

    public ObjectNode toJson() {
        ObjectNode testingDetailsObject = mapper.createObjectNode();
        testingDetailsObject.put("schema-version", schemaVersion);
        testingDetailsObject.put("testing-set-size", testingSetSize);

        ObjectNode testingMetricsObject = mapper.createObjectNode();
        for (String description : metrics.keySet()) {
            testingMetricsObject.put(description, metrics.get(description));
        }
        testingDetailsObject.set("metrics", testingMetricsObject);

        return testingDetailsObject;
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }
}

