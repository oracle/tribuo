import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.tribuo.Model;
import org.tribuo.evaluation.Evaluation;
import org.tribuo.interop.ExternalModel;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class ModelCard {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final ModelDetails modelDetails;
    private final TrainingDetails trainingDetails;
    private final TestingDetails testingDetails;
    private final UsageDetails usageDetails;

    public ModelCard(Model<?> model, Evaluation<?> evaluation) {
        if (model instanceof ExternalModel) throw new IllegalArgumentException();
        modelDetails = new ModelDetails(model);
        trainingDetails = new TrainingDetails(model);
        testingDetails = new TestingDetails(evaluation);
        usageDetails = new UsageDetails();
    }

    public ModelCard(String sourceFile) throws IOException {
        JsonNode modelCardJson = mapper.readTree(Paths.get(sourceFile).toFile());
        modelDetails = new ModelDetails(modelCardJson.get("ModelDetails"));
        trainingDetails = new TrainingDetails(modelCardJson.get("TrainingDetails"));
        testingDetails = new TestingDetails(modelCardJson.get("TestingDetails"));
        usageDetails = new UsageDetails(modelCardJson.get("UsageDetails"));
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
        modelCardObject.set("UsageDetails", usageDetails.toJson());
        return modelCardObject;
    }

    public void saveToFile(String destinationFile) throws IOException {
        ObjectNode modelCardObject = toJson();
        mapper.writeValue(new File(destinationFile), modelCardObject);
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }
}
