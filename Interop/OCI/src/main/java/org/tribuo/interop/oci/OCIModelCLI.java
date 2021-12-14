/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.oci;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.ser.FilterProvider;
import com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider;
import com.oracle.bmc.ConfigFileReader;
import com.oracle.bmc.auth.AuthenticationDetailsProvider;
import com.oracle.bmc.auth.ConfigFileAuthenticationDetailsProvider;
import com.oracle.bmc.datascience.DataScienceClient;
import com.oracle.bmc.http.internal.ExplicitlySetFilter;
import com.oracle.bmc.http.signing.RequestSigningFilter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.Trainer;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.baseline.DummyClassifierTrainer;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.math.la.DenseVector;
import org.tribuo.ONNXExportable;
import org.tribuo.util.Util;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.Invocation;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This class provides a CLI for deploying and scoring a Tribuo Classification model.
 * <p>
 * Deploying other kinds of Tribuo model is similarly straightforward and we may extend this class to do
 * so in the future.
 */
public abstract class OCIModelCLI {
    private static final Logger logger = Logger.getLogger(OCIModelCLI.class.getName());

    /**
     * Private constructor for abstract class.
     */
    private OCIModelCLI() {}

    private static void createModelAndDeploy(OCIModelOptions options) throws IOException, ClassNotFoundException {
        // Load the Tribuo model
        Model<Label> model;
        try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(options.modelPath))) {
            model = Model.castModel((Model<?>) ois.readObject(),Label.class);
        }
        if (!(model instanceof ONNXExportable)) {
            throw new IllegalArgumentException("Model not ONNXExportable, received " + model.toString());
        }

        ObjectMapper mapper = OCIUtil.createObjectMapper();

        // Instantiate the client
        final ConfigFileReader.ConfigFile configFile = ConfigFileReader.parseDefault();
        final AuthenticationDetailsProvider provider = new ConfigFileAuthenticationDetailsProvider(configFile);
        DataScienceClient client = new DataScienceClient(provider);

        OCIUtil.OCIDSConfig dsConfig = new OCIUtil.OCIDSConfig(options.compartmentID,options.projectID);
        OCIUtil.OCIModelArtifactConfig config = new OCIUtil.OCIModelArtifactConfig(dsConfig,options.modelDisplayName,"Deployed Tribuo Model","org.tribuo.oci",1);

        String modelId = OCIUtil.createModel((Model<Label> & ONNXExportable) model,client,mapper,config);

        OCIUtil.OCIModelDeploymentConfig mdConfig = new OCIUtil.OCIModelDeploymentConfig(dsConfig,modelId,options.modelDisplayName,options.instanceShape,options.bandwidth,options.instanceCount);

        OCIUtil.deploy(mdConfig,client,mapper);

        client.close();
    }

    /**
     * Deploy CLI command.
     * @param options CLI options.
     * @throws IOException If the deployment failed.
     */
    private static void deploy(OCIModelOptions options) throws IOException {
        // Instantiate the client
        final ConfigFileReader.ConfigFile configFile = ConfigFileReader.parseDefault();
        final AuthenticationDetailsProvider provider = new ConfigFileAuthenticationDetailsProvider(configFile);
        DataScienceClient client = new DataScienceClient(provider);

        // Setup object mapper for writing to the terminal
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        FilterProvider filters =
                new SimpleFilterProvider()
                        .addFilter(ExplicitlySetFilter.NAME, ExplicitlySetFilter.INSTANCE);
        mapper.setFilterProvider(filters);

        OCIUtil.OCIDSConfig dsConfig = new OCIUtil.OCIDSConfig(options.compartmentID,options.projectID);
        OCIUtil.OCIModelDeploymentConfig config = new OCIUtil.OCIModelDeploymentConfig(dsConfig,options.modelId,options.modelDisplayName,options.instanceShape,options.bandwidth,options.instanceCount);

        String url = OCIUtil.deploy(config,client,mapper);
        System.out.println("Deployment URL = " + url);

        client.close();
    }

    /**
     * Scoring CLI command.
     * @param options CLI options.
     * @throws IOException If the dataset could not be loaded, or the model deployment responded incorrectly.
     * @throws ClassNotFoundException If the dataset has an unknown class inside.
     */
    private static void modelScoring(OCIModelOptions options) throws IOException, ClassNotFoundException {
        // Load the dataset
        Dataset<Label> dataset;
        try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(options.datasetPath))) {
            dataset = Dataset.castDataset((Dataset<?>) ois.readObject(), Label.class);
        }
        ImmutableFeatureMap featureIDMap = dataset.getFeatureIDMap();

        if (options.useOCIModel) {
            // Prep mappings
            Map<String, Integer> featureMapping = new HashMap<>();
            for (VariableInfo f : featureIDMap){
                VariableIDInfo id = (VariableIDInfo) f;
                featureMapping.put(id.getName(),id.getID());
            }
            Map<Label, Integer> outputMapping = new HashMap<>();
            for (Pair<Integer,Label> l : dataset.getOutputIDInfo()) {
                outputMapping.put(l.getB(), l.getA());
            }
            OCIModel<Label> model = OCIModel.createOCIModel(dataset.getOutputFactory(),featureMapping,outputMapping,options.ociConfigFile,options.endpointDomain + options.modelDeploymentId, new OCILabelConverter(true));

            System.out.println("Scoring using OCIModel - " + model.toString());
            LabelEvaluator eval = new LabelEvaluator();
            long startTime = System.currentTimeMillis();
            LabelEvaluation evaluation = eval.evaluate(model,dataset);
            long endTime = System.currentTimeMillis();
            System.out.println("Scoring took - " + Util.formatDuration(startTime,endTime));
            System.out.println((((double) endTime - startTime) / dataset.size()) + "ms per example");

            System.out.println(evaluation.toString());
            System.out.println(evaluation.getConfusionMatrix().toString());
        } else {
            manualModelScoring(options,dataset,featureIDMap);
        }
    }

    private static void manualModelScoring(OCIModelOptions options, Dataset<Label> dataset, ImmutableFeatureMap featureIDMap) throws IOException {
        OCILabelConverter labelConverter = new OCILabelConverter(true);

        // Setup object mapper for parsing the output
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        FilterProvider filters =
                new SimpleFilterProvider()
                        .addFilter(ExplicitlySetFilter.NAME, ExplicitlySetFilter.INSTANCE);
        mapper.setFilterProvider(filters);

        // Pre-Requirement: Allow setting of restricted headers. This is required to allow the SigningFilter
        // to set the host header that gets computed during signing of the request.
        System.setProperty("sun.net.http.allowRestrictedHeaders", "true");

        // 1) Create a request signing filter instance
        RequestSigningFilter requestSigningFilter = RequestSigningFilter.fromConfigFile(
                "~/.oci/config",
                "DEFAULT"
        );

        // 2) Create a Jersey client and register the request signing filter
        Client client = ClientBuilder
                .newBuilder()
                .build()
                .register(requestSigningFilter);

        // 3) Target an endpoint. You must ensure that path arguments and query
        // params are escaped correctly yourself
        WebTarget target = client
                .target(options.endpointDomain + options.modelDeploymentId)
                .path("predict");

        System.out.println("Scoring using manual conversion from endpoint " + options.modelDeploymentId);
        long startTime = System.currentTimeMillis();
        List<Prediction<Label>> predictions = new ArrayList<>();
        for (Example<Label> e : dataset) {
            double[] features = DenseVector.createDenseVector(e, featureIDMap, false).toArray();
            // 4) Set the expected type and invoke the call
            Invocation.Builder ib = target.request();
            ib.accept(MediaType.APPLICATION_JSON);
            Response response = ib.buildPost(Entity.entity("[" + Arrays.toString(features) + "]", MediaType.APPLICATION_JSON)).invoke();

            String json;
            try (BufferedReader reader = new BufferedReader(new InputStreamReader((InputStream) response.getEntity(), StandardCharsets.UTF_8))) {
                StringBuilder jsonBody = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    jsonBody.append(line);
                }
                json = jsonBody.toString();
            } catch (IOException ex) {
                throw new IllegalStateException("Failed to read response from input stream", ex);
            }
            try {
                PredictionJson predJson = mapper.readValue(json, PredictionJson.class);
                predictions.add(labelConverter.convertOutput(DenseVector.createDenseVector(predJson.prediction[0]), features.length, e, dataset.getOutputIDInfo()));
            } catch (JsonParseException | JsonMappingException ex) {
                // Must be an error condition
                logger.log(Level.WARNING, "Failed to parse Json, '" + json + "' - " + ex.getMessage());
            }
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Scoring took - " + Util.formatDuration(startTime,endTime));
        System.out.println((((double) endTime - startTime) / dataset.size()) + "ms per example");

        // Make a fake model to make the evaluator happy
        Trainer<Label> trainer = DummyClassifierTrainer.createUniformTrainer(1L);
        Model<Label> fakeModel = trainer.train(dataset);

        LabelEvaluator eval = new LabelEvaluator();
        LabelEvaluation evaluation = eval.evaluate(fakeModel,predictions,dataset.getProvenance());

        System.out.println(evaluation.toString());
        System.out.println(evaluation.getConfusionMatrix().toString());
    }

    /**
     * Carrier type for easy deserialization from JSON.
     */
    public static final class PredictionJson {
        @JsonProperty("prediction")
        public double[][] prediction;

        @JsonCreator
        public PredictionJson(@JsonProperty("prediction") double[][] prediction) {
            this.prediction = prediction;
        }
    }

    /**
     * Entry point for the OCIModelCLI.
     * @param args The CLI arguments.
     * @throws IOException If the model or dataset failed to load.
     * @throws ClassNotFoundException If the model or dataset had an unknown class.
     */
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        OCIModelOptions o = new OCIModelOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o,false);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        switch (o.mode) {
            case CREATE_AND_DEPLOY:
                createModelAndDeploy(o);
                break;
            case DEPLOY:
                deploy(o);
                break;
            case SCORE:
                modelScoring(o);
                break;
        }

    }

    /**
     * Options for the OCIModelCLI.
     */
    public static final class OCIModelOptions implements Options {
        /**
         * Mode for the CLI.
         */
        public enum Mode {
            /**
             * Create a Model artifact, upload it to OCI and create a Model Deployment.
             */
            CREATE_AND_DEPLOY,
            /**
             * Create a Model deployment.
             */
            DEPLOY,
            /**
             * Score a deployed model.
             */
            SCORE
        };
        @Override
        public String getOptionsDescription() {
            return "OCIModelCLI deploys and scores a Tribuo Classification model using OCI Data Science Model Deployment.";
        }

        // Options for both commands
        /**
         * Deploy or score an OCI DS model.
         */
        @Option(charName='m',longName="mode",usage="Deploy or score an OCI DS model.")
        public Mode mode;
        /**
         * Project ID.
         */
        @Option(charName='p',longName="project-id",usage="Project ID.")
        public String projectID;
        /**
         * Compartment ID.
         */
        @Option(charName='c',longName="compartment-id",usage="Compartment ID.")
        public String compartmentID;
        // Deployment options
        /**
         * Path to the serialized model to deploy to OCI DS.
         */
        @Option(charName='d',longName="deploy-model-path",usage="Path to the serialized model to deploy to OCI DS.")
        public Path modelPath;
        /**
         * Model display name.
         */
        @Option(longName="model-display-name",usage="Model display name.")
        public String modelDisplayName = "tribuo-test";
        /**
         * Number of model instances to deploy.
         */
        @Option(longName="model-instance-count",usage="Number of model instances to deploy.")
        public int instanceCount = 1;
        /**
         * Model bandwidth in MBps.
         */
        @Option(longName="model-bandwidth",usage="Model bandwidth in MBPS.")
        public int bandwidth = 10;
        /**
         * OCI shape to run the model on.
         */
        @Option(longName="model-instance-shape",usage="OCI shape to run the model on.")
        public String instanceShape = "VM.Standard2.1";
        /**
         * The id of the model.
         */
        @Option(longName="model-id",usage="The id of the model.")
        public String modelId;
        // Scoring options
        /**
         * Path to the serialized dataset to score.
         */
        @Option(charName='s',longName="dataset-path",usage="Path to the serialized dataset to score.")
        public Path datasetPath;
        /**
         * The id of the model deployment.
         */
        @Option(charName='i',longName="model-deployment-id",usage="The id of the model deployment.")
        public String modelDeploymentId;
        /**
         * The OCI endpoint domain.
         */
        @Option(longName="oci-domain",usage="The OCI endpoint domain.")
        public String endpointDomain;
        /**
         * Use the OCIModel class.
         */
        @Option(longName="oci-model",usage="Use the OCIModel class.")
        public boolean useOCIModel;
        /**
         * OCI config file path for the OCIModel class.
         */
        @Option(longName="oci-config-file",usage="OCI config file path for the OCIModel class.")
        public Path ociConfigFile;
    }
}
