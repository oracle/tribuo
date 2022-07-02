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

public final class TestingDetails {
    private static final String schemaVersion = "1.0";
    private final int testingSetSize;
    private final Map<String, Double> metrics = new HashMap<>();

    public TestingDetails(Evaluation<?> evaluation) {
        testingSetSize = evaluation.getPredictions().size();
    }

    public TestingDetails(Evaluation<?> evaluation, Map<String, Double> testingMetrics) {
        testingSetSize = evaluation.getPredictions().size();
        metrics.putAll(testingMetrics);
    }

    public TestingDetails(JsonNode testingDetailsJson) throws JsonProcessingException {
        testingSetSize = testingDetailsJson.get("testing-set-size").intValue();
        Map<?, ?> parsed = mapper.readValue(testingDetailsJson.get("metrics").toString(), Map.class);
        for (Map.Entry<?,?> entry : parsed.entrySet()) {
            metrics.put((String) entry.getKey(), (Double) entry.getValue());
        }

    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public int getTestingSetSize() {
        return testingSetSize;
    }

    public Map<String, Double> getMetrics() {
        return Collections.unmodifiableMap(metrics);
    }

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

