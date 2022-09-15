/*
 * Copyright (c) 2021, 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.ser.FilterProvider;
import com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider;
import com.oracle.bmc.datascience.DataScienceClient;
import com.oracle.bmc.http.internal.ExplicitlySetFilter;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.VariableIDInfo;
import org.tribuo.VariableInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.evaluation.LabelEvaluation;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.ONNXExportable;
import org.tribuo.util.Util;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
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
        if (options.modelProtobuf) {
            model = Model.deserializeFromFile(options.modelPath).castModel(Label.class);
        } else {
            try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(options.modelPath))) {
                model = ((Model<?>)ois.readObject()).castModel(Label.class);
            }
        }
        if (!(model instanceof ONNXExportable)) {
            throw new IllegalArgumentException("Model not ONNXExportable, received " + model.toString());
        }

        ObjectMapper mapper = OCIUtil.createObjectMapper();

        // Instantiate the client & configurations
        DataScienceClient client = options.makeClient();
        OCIUtil.OCIDSConfig dsConfig = new OCIUtil.OCIDSConfig(options.compartmentID,options.projectID);
        OCIUtil.OCIModelArtifactConfig config = new OCIUtil.OCIModelArtifactConfig(dsConfig,options.modelDisplayName,"Deployed Tribuo Model","org.tribuo.oci",1,options.condaName,options.condaPath);

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
        DataScienceClient client = options.makeClient();

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
        if (options.datasetProtobuf) {
            dataset = Dataset.castDataset(Dataset.deserializeFromFile(options.datasetPath), Label.class);
        } else {
            try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(options.datasetPath))) {
                dataset = Dataset.castDataset((Dataset<?>) ois.readObject(), Label.class);
            }
        }
        ImmutableFeatureMap featureIDMap = dataset.getFeatureIDMap();

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
        OCIModel<Label> model = OCIModel.createOCIModel(dataset.getOutputFactory(),featureMapping,outputMapping,
                options.ociConfigFile,options.ociConfigProfile,
                options.endpointDomain + options.modelDeploymentId,
                new OCILabelConverter(true));

        System.out.println("Scoring using OCIModel - " + model.toString());
        LabelEvaluator eval = new LabelEvaluator();
        long startTime = System.currentTimeMillis();
        LabelEvaluation evaluation = eval.evaluate(model,dataset);
        long endTime = System.currentTimeMillis();
        System.out.println("Scoring took - " + Util.formatDuration(startTime,endTime));
        System.out.println((((double) endTime - startTime) / dataset.size()) + "ms per example");

        System.out.println(evaluation.toString());
        System.out.println(evaluation.getConfusionMatrix().toString());
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
         * Is the model stored in protobuf format?
         */
        @Option(longName="model-protobuf",usage="Is the model stored in protobuf format?")
        public boolean modelProtobuf;
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
         * Is the serialized dataset in protobuf format?
         */
        @Option(longName="dataset-protobuf",usage="Is the serialized dataset a protobuf?")
        public boolean datasetProtobuf;
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
         * OCI config file path.
         */
        @Option(longName="oci-config-file",usage="OCI config file path. If null use the default.")
        public Path ociConfigFile = null;
        /**
         * OCI profile in the config file.
         */
        @Option(longName="oci-config-file-profile",usage="OCI profile in the config file. If null use the default.")
        public String ociConfigProfile = null;
        /**
         * OCI DS conda environment name.
         */
        @Option(longName="conda-name",usage="OCI DS conda environment name.")
        public String condaName;
        /**
         * OCI DS conda environment path in object storage.
         */
        @Option(longName="conda-name",usage="OCI DS conda environment path in object storage.")
        public String condaPath;

        /**
         * Makes the DataScienceClient specified by these options.
         * @return The DataScienceClient.
         * @throws IOException If the config file could not be read.
         */
        DataScienceClient makeClient() throws IOException {
            return new DataScienceClient(OCIModel.makeAuthProvider(ociConfigFile, ociConfigProfile));
        }

    }
}
