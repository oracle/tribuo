import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.oracle.labs.mlrg.olcut.command.Command;
import com.oracle.labs.mlrg.olcut.command.CommandGroup;
import com.oracle.labs.mlrg.olcut.command.CommandInterpreter;
import org.tribuo.ModelExplorer;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class UsageDetails implements CommandGroup {
    private static final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);;
    private static final Logger logger = Logger.getLogger(ModelExplorer.class.getName());
    private final CommandInterpreter shell = new CommandInterpreter();
    private final String schemaVersion;
    private String intendedUse;
    private String intendedUsers;
    private final List<String> outOfScopeUses = new ArrayList<>();
    private final List<String> preProcessingSteps = new ArrayList<>();
    private final List<String> considerations = new ArrayList<>();
    private final List<String> factors = new ArrayList<>();
    private final List<String> resources = new ArrayList<>();
    private String primaryContact = null;
    private String modelCitation = null;
    private String modelLicense = null;

    public UsageDetails() {
        schemaVersion = "1.0";
        intendedUse = "";
        intendedUsers = "";
        primaryContact = "";
        modelCitation = "";
        modelLicense = "";
        shell.setPrompt("CLI% ");
    }

    public UsageDetails(JsonNode usageDetailsJson) {
        schemaVersion = usageDetailsJson.get("schema-version").textValue();
        intendedUse = usageDetailsJson.get("intended-use").textValue();
        intendedUsers = usageDetailsJson.get("intended-users").textValue();

        for (int i = 0; i < usageDetailsJson.get("out-of-scope-uses").size(); i++)
            outOfScopeUses.add(usageDetailsJson.get("out-of-scope-uses").get(i).textValue());
        for (int i = 0; i < usageDetailsJson.get("pre-processing-steps").size(); i++)
            preProcessingSteps.add(usageDetailsJson.get("pre-processing-steps").get(i).textValue());
        for (int i = 0; i < usageDetailsJson.get("considerations-list").size(); i++)
            considerations.add(usageDetailsJson.get("considerations-list").get(i).textValue());
        for (int i = 0; i < usageDetailsJson.get("relevant-factors-list").size(); i++)
            factors.add(usageDetailsJson.get("relevant-factors-list").get(i).textValue());
        for (int i = 0; i < usageDetailsJson.get("resources-list").size(); i++)
            resources.add(usageDetailsJson.get("resources-list").get(i).textValue());

        primaryContact = usageDetailsJson.get("primary-contact").textValue();
        modelCitation = usageDetailsJson.get("model-citation").textValue();
        modelLicense = usageDetailsJson.get("model-license").textValue();
    }

    public String getSchemaVersion() {
        return schemaVersion;
    }

    public String getIntendedUse() {
        return intendedUse;
    }

    public String getIntendedUsers() {
        return intendedUsers;
    }

    public List<String> getOutOfScopeUses() {
        return outOfScopeUses;
    }

    public List<String> getPreProcessingSteps() {
        return preProcessingSteps;
    }

    public List<String> getConsiderations() {
        return considerations;
    }

    public List<String> getFactors() {
        return factors;
    }

    public List<String> getResources() {
        return resources;
    }

    public String getPrimaryContact() {
        return primaryContact;
    }

    public String getModelCitation() {
        return modelCitation;
    }

    public String getModelLicense() {
        return modelLicense;
    }

    public void startShell() {
        shell.add(this);
        shell.start();
    }

    public ObjectNode toJson() {
        ObjectNode usageDetailsObject = mapper.createObjectNode();
        usageDetailsObject.put("schema-version", schemaVersion);
        usageDetailsObject.put("intended-use", intendedUse);
        usageDetailsObject.put("intended-users", intendedUsers);

        ArrayNode usesArr = mapper.createArrayNode();
        for (String s : outOfScopeUses) usesArr.add(s);
        usageDetailsObject.set("out-of-scope-uses", usesArr);

        ArrayNode processingArr = mapper.createArrayNode();
        for (String s : preProcessingSteps) processingArr.add(s);
        usageDetailsObject.set("pre-processing-steps", processingArr);

        ArrayNode considerationsArr = mapper.createArrayNode();
        for (String s : considerations) considerationsArr.add(s);
        usageDetailsObject.set("considerations-list", considerationsArr);

        ArrayNode factorsArr = mapper.createArrayNode();
        for (String s : factors) factorsArr.add(s);
        usageDetailsObject.set("relevant-factors-list", factorsArr);

        ArrayNode resourcesArr = mapper.createArrayNode();
        for (String s : resources) resourcesArr.add(s);
        usageDetailsObject.set("resources-list", resourcesArr);

        usageDetailsObject.put("primary-contact", primaryContact);
        usageDetailsObject.put("model-citation", modelCitation);
        usageDetailsObject.put("model-license", modelLicense);

        return usageDetailsObject;
    }

    @Override
    public String getName() {
        return "UsageDetails";
    }

    @Override
    public String getDescription() {
        return "Commands for specifying UsageDetails for a model.";
    }

    @Command(
            usage = "<String> Records intended use of model."
    )
    public String intendedUse(CommandInterpreter ci, String use) {
        intendedUse = use;
        return("Recorded intended use as " + intendedUse + ".");
    }

    @Command(
            usage = "<String> Records intended users of model."
    )
    public String intendedUsers(CommandInterpreter ci, String users) {
        intendedUsers = users;
        return("Recorded intended users as " + intendedUsers + ".");
    }

    @Command(
            usage = "<String> Adds an out-of-scope use to list of out-of-scope uses."
    )
    public String addOutOfScopeUse(CommandInterpreter ci, String use) {
        outOfScopeUses.add(use);
        return("Added an out-of-scope use to list of out-of-scope uses.");
    }

    @Command(
            usage = "<int> Remove out-of-scope use at specified index (1-indexed)."
    )
    public String removeOutOfScopeUse(CommandInterpreter ci, int index) {
        outOfScopeUses.remove(index-1);
        return("Removed out-of-scope use at specified index.");
    }

    @Command(
            usage = "Displays all added out-of-scope uses."
    )
    public String viewOutOfScopeUse(CommandInterpreter ci) {
        for (int i = 0; i < outOfScopeUses.size(); i++) {
            System.out.println("\t" + (i+1) + ") "+ outOfScopeUses.get(i));
        }
        return("Displayed all added out-of-scope uses.");
    }

    @Command(
            usage = "<String> Adds pre-processing step to list of steps."
    )
    public String addPreProcessingStep(CommandInterpreter ci, String step) {
        preProcessingSteps.add(step);
        return("Added pre-processing step to list of steps.");
    }

    @Command(
            usage = "<int> Remove pro-processing step at specified index (1-indexed)."
    )
    public String removePreProcessingStep(CommandInterpreter ci, int index) {
        preProcessingSteps.remove(index-1);
        return("Removed pre-processing step at specified index.");
    }

    @Command(
            usage = "Displays all added pre-processing steps."
    )
    public String viewPreProcessingSteps(CommandInterpreter ci) {
        for (int i = 0; i < preProcessingSteps.size(); i++) {
            System.out.println("\t" + (i+1) + ") "+ preProcessingSteps.get(i));
        }
        return("Displayed all added pre-processing steps.");
    }

    @Command(
            usage = "<String> Adds consideration to list of considerations."
    )
    public String addConsideration(CommandInterpreter ci, String consideration) {
        considerations.add(consideration);
        return("Added consideration to list of considerations.");
    }

    @Command(
            usage = "<int> Remove consideration at specified index (1-indexed)."
    )
    public String removeConsideration(CommandInterpreter ci, int index) {
        considerations.remove(index-1);
        return("Removed consideration at specified index.");
    }

    @Command(
            usage = "Displays all added considerations."
    )
    public String viewConsiderations(CommandInterpreter ci) {
        for (int i = 0; i < considerations.size(); i++) {
            System.out.println("\t" + (i+1) + ") "+ considerations.get(i));
        }
        return("Displayed all added considerations.");
    }

    @Command(
            usage = "<String> Adds relevant factor to list of factors."
    )
    public String addFactor(CommandInterpreter ci, String factor) {
        factors.add(factor);
        return("Added factor to list of factors.");
    }

    @Command(
            usage = "<int> Remove factor at specified index (1-indexed)."
    )
    public String removeFactor(CommandInterpreter ci, int index) {
        factors.remove(index-1);
        return("Removed factor at specified index.");
    }

    @Command(
            usage = "Displays all added factors."
    )
    public String viewFactors(CommandInterpreter ci) {
        for (int i = 0; i < factors.size(); i++) {
            System.out.println("\t" + (i+1) + ") "+ factors.get(i));
        }
        return("Displayed all added factors.");
    }

    @Command(
            usage = "<String> Adds resource to list of resources."
    )
    public String addResource(CommandInterpreter ci, String resource) {
        resources.add(resource);
        return("Added resource to list of resources.");
    }

    @Command(
            usage = "<int> Remove resource at specified index (1-indexed)."
    )
    public String removeResource(CommandInterpreter ci, int index) {
        resources.remove(index-1);
        return("Removed resource at specified index.");
    }

    @Command(
            usage = "Displays all added resources."
    )
    public String viewResources(CommandInterpreter ci) {
        for (int i = 0; i < resources.size(); i++) {
            System.out.println("\t" + (i+1) + ") "+ resources.get(i));
        }
        return("Displayed all added resources.");
    }

    @Command(
            usage = "<String> Records primary contact in case of questions or comments."
    )
    public String primaryContact(CommandInterpreter ci, String contact) {
        primaryContact = contact;
        return("Recorded primary contact as " + primaryContact + ".");
    }

    @Command(
            usage = "<String> Records model's citation."
    )
    public String modelCitation(CommandInterpreter ci, String citation) {
        modelCitation = citation;
        return("Recorded model citation as " + modelCitation + ".");
    }

    @Command(
            usage = "<String> Records model's license."
    )
    public String modelLicense(CommandInterpreter ci, String license) {
        modelLicense = license;
        return("Recorded model license as " + modelLicense + ".");
    }

    @Command(
            usage = "<String> Saves UsageDetails to provided destination file and closes CLI."
    )
    public String saveUsageDetails(CommandInterpreter ci, String destinationFile) throws IOException {
        ObjectNode modelCardObject = (ObjectNode)mapper.readTree(Paths.get(destinationFile).toFile());
        ObjectNode usageDetailsObject = toJson();
        modelCardObject.set("UsageDetails", usageDetailsObject);
        mapper.writeValue(new File(destinationFile), modelCardObject);
        return "Saved UsageDetails to destination file and closing CLI.";
    }

    @Command(
            usage = "Closes shell without saving any recorded content."
    )
    public String close(CommandInterpreter ci) {
        shell.close();
        return "Closed shell.";
    }

    @Override
    public String toString() {
        return toJson().toPrettyString();
    }

    public static void main(String[] args) {
        UsageDetails driver = new UsageDetails();
        driver.startShell();
    }
}