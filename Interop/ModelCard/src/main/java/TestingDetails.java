import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.evaluation.Evaluation;

import java.util.HashMap;
import java.util.Map;

public class TestingDetails {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final String schemaVersion;
    private final int testingSetSize;
    private final Map<String, Double> metrics = new HashMap<>();

    public TestingDetails(Evaluation<?> evaluation) {
        schemaVersion = "1.0";
        testingSetSize = evaluation.getPredictions().size();
    }

    public TestingDetails(JsonNode testingDetailsJson) throws JsonProcessingException {
        schemaVersion = testingDetailsJson.get("schema-version").textValue();
        testingSetSize = testingDetailsJson.get("testing-set-size").intValue();
        Map<String, Double> parsed = mapper.readValue(testingDetailsJson.get("metrics").toString(), Map.class);
        for (var entry : parsed.keySet())
            metrics.put(entry, parsed.get(entry));
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

