import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.Model;
import org.tribuo.ensemble.EnsembleModel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrainingDetails {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);;
    private final String schemaVersion;
    private final String trainingTime;
    private final int trainingSetSize;
    private final int numFeatures;
    private final List<String> features = new ArrayList<>();;
    private final int numOutputs;
    private final Map<String, Long> outputsDistribution = new HashMap<>();

    public TrainingDetails(Model<?> model) {
        schemaVersion = "1.0";
        trainingTime = model.getProvenance().getTrainingTime().toString();
        trainingSetSize = model.getProvenance().getDatasetProvenance().getNumExamples();

        numFeatures = model.getProvenance().getDatasetProvenance().getNumFeatures();
        for (int i = 0; i < model.getFeatureIDMap().size(); i++)
            features.add(model.getFeatureIDMap().get(i).getName());

        numOutputs = model.getProvenance().getDatasetProvenance().getNumOutputs();

        boolean includeOutputDist = true;
        if (model instanceof EnsembleModel) {
            EnsembleModel<?> ensemble = (EnsembleModel<?>) model;
            for (var m : ensemble.getModels()) {
                if (m.getClass().getTypeName().contains("regression")) {
                    includeOutputDist = false;
                    break;
                }
            }
        } else if (model.getClass().getTypeName().contains("regression"))
            includeOutputDist = false;

        if (includeOutputDist) {
            for (var pair : model.getOutputIDInfo().outputCountsIterable())
                outputsDistribution.put(pair.getA(), pair.getB());
        }
    }

    public TrainingDetails(JsonNode trainingDetailsJson) throws JsonProcessingException {
        schemaVersion = trainingDetailsJson.get("schema-version").textValue();
        trainingTime = trainingDetailsJson.get("training-time").textValue();
        trainingSetSize = trainingDetailsJson.get("training-set-size").intValue();

        numFeatures = trainingDetailsJson.get("num-features").intValue();
        for (int i = 0; i < trainingDetailsJson.get("features-list").size(); i++)
            features.add(trainingDetailsJson.get("features-list").get(i).textValue());

        numOutputs = trainingDetailsJson.get("num-outputs").intValue();
        Map<String, Integer> parsed = mapper.readValue(trainingDetailsJson.get("outputs-distribution").toString(), Map.class);
        for (var entry : parsed.keySet())
            outputsDistribution.put(entry, parsed.get(entry).longValue());
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
        for (String s : features) featuresArr.add(s);
        datasetDetailsObject.set("features-list", featuresArr);

        datasetDetailsObject.put("num-outputs", numOutputs);
        ObjectNode outputsArr = mapper.createObjectNode();
        for (String description : outputsDistribution.keySet())
            outputsArr.put(description, outputsDistribution.get(description));
        datasetDetailsObject.set("outputs-distribution", outputsArr);

        return datasetDetailsObject;
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }
}