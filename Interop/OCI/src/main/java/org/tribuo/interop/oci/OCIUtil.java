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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.ser.FilterProvider;
import com.fasterxml.jackson.databind.ser.impl.SimpleFilterProvider;
import com.oracle.bmc.datascience.DataScienceClient;
import com.oracle.bmc.datascience.model.CreateModelDeploymentDetails;
import com.oracle.bmc.datascience.model.CreateModelDetails;
import com.oracle.bmc.datascience.model.FixedSizeScalingPolicy;
import com.oracle.bmc.datascience.model.InstanceConfiguration;
import com.oracle.bmc.datascience.model.Metadata;
import com.oracle.bmc.datascience.model.ModelConfigurationDetails;
import com.oracle.bmc.datascience.model.ModelDeployment;
import com.oracle.bmc.datascience.model.SingleModelDeploymentConfigurationDetails;
import com.oracle.bmc.datascience.requests.CreateModelArtifactRequest;
import com.oracle.bmc.datascience.requests.CreateModelDeploymentRequest;
import com.oracle.bmc.datascience.requests.CreateModelRequest;
import com.oracle.bmc.datascience.responses.CreateModelArtifactResponse;
import com.oracle.bmc.http.internal.ExplicitlySetFilter;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Tribuo;
import org.tribuo.anomaly.Event;
import org.tribuo.classification.Label;
import org.tribuo.clustering.ClusterID;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.regression.Regressor;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * Utils for uploading and deploying models to OCI Data Science.
 */
public abstract class OCIUtil {

    private static final Logger logger = Logger.getLogger(OCIUtil.class.getName());

    /**
     * Enum for OCI model types.
     */
    public enum OCIModelType {
        /**
         * Binary classification.
         */
        BINARY_CLASSIFICATION("binary_classification"),
        /**
         * Regression, maps to Tribuo's {@link Regressor}.
         */
        REGRESSION("regression"),
        /**
         * Multi-class classification, maps to Tribuo's {@link Label}.
         */
        MULTINOMIAL_CLASSIFICATION("multinomial_classification"),
        /**
         * Clustering, maps to Tribuo's {@link ClusterID}.
         */
        CLUSTERING("clustering"),
        /**
         * Recommender system, no Tribuo mapping.
         */
        RECOMMENDER("recommender"),
        /**
         * Dimensionality reduction, no Tribuo mapping.
         */
        DIMENSIONALITY_REDUCTION("dimensionality_reduction/representation"),
        /**
         * Time series forecasting, no Tribuo mapping.
         */
        TIME_SERIES("time_series_forecasting"),
        /**
         * Anomaly detection, maps to Tribuo's {@link Event}.
         */
        ANOMALY_DETECTION("anomaly_detection"),
        /**
         * Topic modelling, no Tribuo mapping.
         */
        TOPIC_MODELLING("topic_modelling"),
        /**
         * Named Entity Recognition, no strict Tribuo mapping.
         */
        NER("ner"),
        /**
         * Sentiment analysis, no strict Tribuo mapping.
         */
        SENTIMENT_ANALYSIS("sentiment_analysis"),
        /**
         * Image classification, no strict Tribuo mapping.
         */
        IMAGE_CLASSIFICATION("image_classification"),
        /**
         * Object localization, no Tribuo mapping.
         */
        OBJECT_LOCALIZATION("object_localization"),
        /**
         * Other prediction types, currently used as the mapping for Tribuo's {@link MultiLabel}.
         */
        OTHER("other");

        public final String modelType;

        private OCIModelType(String modelType) {
            this.modelType = modelType;
        }
    }

    /**
     * Configuration for OCI DS.
     * <p>
     * Not a record but will be.
     */
    public static final class OCIDSConfig {
        /**
         * OCI compartment ID.
         */
        public final String compartmentID;
        /**
         * OCI Data Science project ID.
         */
        public final String projectID;

        /**
         * Constructs an OCIDSConfig.
         * @param compartmentID The OCI compartment ID.
         * @param projectID The OCI DS project ID.
         */
        public OCIDSConfig(String compartmentID, String projectID) {
            this.compartmentID = compartmentID;
            this.projectID = projectID;
        }
    }

    /**
     * Configuration for an OCI DS Model artifact.
     * <p>
     * Not a record, but will be.
     */
    public static final class OCIModelArtifactConfig {
        /**
         * The OCI Data Science config.
         */
        public final OCIDSConfig dsConfig;
        /**
         * The model display name.
         */
        public final String modelName;
        /**
         * The model description.
         */
        public final String modelDescription;
        /**
         * The ONNX domain name.
         */
        public final String onnxDomain;
        /**
         * The ONNX model version.
         */
        public final int onnxModelVersion;

        /**
         * Constructs an OCIModelArtifactConfig, used to create an OCI DS model.
         * @param dsConfig The OCI Data Science config.
         * @param modelName The model display name.
         * @param modelDescription The model description.
         * @param onnxDomain The domain to encode in the ONNX file. Should be a reverse-DNS style name, like a Java package.
         * @param onnxModelVersion The ONNX model version number.
         */
        public OCIModelArtifactConfig(OCIDSConfig dsConfig, String modelName, String modelDescription, String onnxDomain, int onnxModelVersion) {
            this.dsConfig = dsConfig;
            this.modelDescription = modelDescription;
            this.modelName = modelName;
            this.onnxDomain = onnxDomain;
            this.onnxModelVersion = onnxModelVersion;
        }
    }

    /**
     * Configuration for an OCI DS Model Deployment.
     * <p>
     * Not a record, but will be.
     */
    public static final class OCIModelDeploymentConfig {
        /**
         * The OCI Data Science config.
         */
        public final OCIDSConfig dsConfig;
        /**
         * The bandwidth for the load balancer in MBps.
         */
        public final int bandwidth;
        /**
         * The number of instances to create.
         */
        public final int instanceCount;
        /**
         * The deployment name.
         */
        public final String deploymentName;
        /**
         * The instance shape.
         */
        public final String shape;
        /**
         * The ID of the model artifact to deploy.
         */
        public final String modelID;

        /**
         * Constructs an OCI DS Model Deployment configuration.
         * @param dsConfig The OCI Data Science config.
         * @param modelID The ID of the model artifact to deploy.
         * @param deploymentName The model deployment name.
         * @param shape The instance shape.
         * @param bandwidth The bandwidth for the load balancer in MBps (minimum value is 10).
         * @param instanceCount The number of instances to spawn.
         */
        public OCIModelDeploymentConfig(OCIDSConfig dsConfig, String modelID, String deploymentName, String shape, int bandwidth, int instanceCount) {
            if (instanceCount < 1) {
                throw new IllegalArgumentException("Instance count must be positive, found " + instanceCount);
            }
            if (bandwidth < 10) {
                throw new IllegalArgumentException("Bandwidth must be 10 or greater, found " + bandwidth);
            }
            if (deploymentName == null || deploymentName.isEmpty()) {
                throw new IllegalArgumentException("Must supply valid deployment name");
            }
            if (shape == null || shape.isEmpty()) {
                throw new IllegalArgumentException("Must supply valid instance shape");
            }
            if (modelID == null || modelID.isEmpty()) {
                throw new IllegalArgumentException("Must supply valid modelID");
            }
            this.dsConfig = dsConfig;
            this.modelID = modelID;
            this.deploymentName = deploymentName;
            this.shape = shape;
            this.bandwidth = bandwidth;
            this.instanceCount = instanceCount;
        }
    }

    /**
     * Private constructor for abstract utility class.
     */
    private OCIUtil() {}

    /**
     * Write a resource to the root of the specified filesystem (usually a Zip file).
     * @param zipFile The filesystem to write to.
     * @param resourceName The resource name to load.
     * @throws IOException If the resource could not be written.
     */
    private static void storeResource(FileSystem zipFile, String resourceName) throws IOException {
        InputStream stream = OCIUtil.class.getResourceAsStream(resourceName);
        storeStream(zipFile,resourceName,stream);
    }

    /**
     * Write a stream into the specified filename in the supplied filesystem.
     * @param zipFile The filesystem to write to.
     * @param filename The filename to use.
     * @param stream The stream to write.
     * @throws IOException If the stream could not be written.
     */
    private static void storeStream(FileSystem zipFile, String filename, InputStream stream) throws IOException {
        Path fileInZip = zipFile.getPath("/",filename);
        Files.copy(stream,fileInZip);
    }

    /**
     * Creates an ObjectMapper capable of parsing the OCI DS json.
     * @return A configured ObjectMapper.
     */
    public static ObjectMapper createObjectMapper() {
        // Setup object mapper for writing to the terminal
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        FilterProvider filters =
                new SimpleFilterProvider()
                        .addFilter(ExplicitlySetFilter.NAME, ExplicitlySetFilter.INSTANCE);
        mapper.setFilterProvider(filters);

        return mapper;
    }

    /**
     * Creates an OCI DS model and uploads the model artifact.
     * <p>
     * Uses ORT as the deployment environment, and inserts a suitable {@code score.py} and {@code runtime.yaml} which
     * assume the input is a plain multi-dimensional json array of floats, and the output is a json object with a
     * single field called "predictions".
     * @param model The model to deploy.
     * @param client The DS Client to handle authentication.
     * @param mapper The object mapper for writing out the JSON results.
     * @param config The upload configuration.
     * @param <T> The model output type.
     * @param <U> The model class, must be {@link ONNXExportable}.
     * @return The model ID.
     * @throws IOException If the zip file could not be created, or it would not upload to OCI DS.
     */
    public static <T extends Output<T>, U extends Model<T> & ONNXExportable> String createModel(U model, DataScienceClient client, ObjectMapper mapper, OCIModelArtifactConfig config) throws IOException {
        // Export the model to ONNX
        Path modelFile = Files.createTempFile("model",".onnx");
        model.saveONNXModel(config.onnxDomain,config.onnxModelVersion,modelFile);
        modelFile.toFile().deleteOnExit();

        ModelProvenance provenance = model.getProvenance();

        OCIModelType modelType;
        if (model.validate(Label.class)) {
            modelType = OCIModelType.MULTINOMIAL_CLASSIFICATION;
        } else if (model.validate(Regressor.class)) {
            modelType = OCIModelType.REGRESSION;
        } else if (model.validate(Event.class)) {
            modelType = OCIModelType.ANOMALY_DETECTION;
        } else if (model.validate(MultiLabel.class)) {
            modelType = OCIModelType.OTHER;
        } else if (model.validate(ClusterID.class)) {
            modelType = OCIModelType.CLUSTERING;
        } else {
            throw new IllegalArgumentException("Unsupported model type " + model.toString());
        }

        return createModel(modelFile,provenance,modelType,client,mapper,config);
    }

    /**
     * Creates an OCI DS model and uploads the model artifact.
     * <p>
     * Uses ORT as the deployment environment, and inserts a suitable {@code score.py} and {@code runtime.yaml} which
     * assume the input is a plain multi-dimensional json array of floats, and the output is a json object with a
     * single field called "predictions".
     * @param onnxFile The ONNX file to deploy.
     * @param provenance The model provenance for storing in the model catalog.
     * @param modelType The model type.
     * @param client The DS Client to handle authentication.
     * @param mapper The object mapper for writing out the JSON results.
     * @param config The upload configuration.
     * @return The model ID.
     * @throws IOException If the zip file could not be created, or it would not upload to OCI DS.
     */
    public static String createModel(Path onnxFile, ModelProvenance provenance, OCIModelType modelType, DataScienceClient client, ObjectMapper mapper, OCIModelArtifactConfig config) throws IOException {
        // Create model artifact
        Path zipFile = Files.createTempFile("oci-ds-model-deployment",".zip");
        try (FileSystem zipFS = FileSystems.newFileSystem(zipFile.toUri(), Collections.emptyMap())) {
            OCIUtil.storeResource(zipFS,"score.py");
            OCIUtil.storeResource(zipFS,"runtime.yaml");
            OCIUtil.storeStream(zipFS,"model.onnx",Files.newInputStream(onnxFile));
        }
        zipFile.toFile().deleteOnExit();

        // Create the model metadata
        CreateModelDetails.Builder modelDetailsBuilder = CreateModelDetails.builder();
        List<Metadata> metadata = new ArrayList<>();
        metadata.add(Metadata.builder().key("UseCaseType").value(modelType.modelType).build());
        metadata.add(Metadata.builder().key("Framework").value("other").build());
        metadata.add(Metadata.builder().key("FrameworkVersion").value(Tribuo.VERSION).build());
        metadata.add(Metadata.builder().key("Algorithm").value(provenance.getTrainerProvenance().getClassName()).build());
        metadata.add(Metadata.builder().key("hyperparameters").value(mapper.writeValueAsString(ProvenanceUtil.convertToMap(provenance.getTrainerProvenance()))).build());
        modelDetailsBuilder.definedMetadataList(metadata);
        modelDetailsBuilder.compartmentId(config.dsConfig.compartmentID);
        modelDetailsBuilder.projectId(config.dsConfig.projectID);
        modelDetailsBuilder.displayName(config.modelName);
        modelDetailsBuilder.description(config.modelDescription);

        // Create the model
        CreateModelRequest.Builder createModelRequest = CreateModelRequest.builder();
        createModelRequest.createModelDetails(modelDetailsBuilder.build());
        com.oracle.bmc.datascience.model.Model creationResponse = client.createModel(createModelRequest.build()).getModel();
        String creationResponseStr = mapper.writeValueAsString(creationResponse);
        logger.info("\n\nCreate model response: \n"+creationResponseStr);
        String modelId = creationResponse.getId();
        logger.info("Model ID = " + modelId);

        // Upload the artifact
        CreateModelArtifactRequest.Builder createArtifactRequest = CreateModelArtifactRequest.builder();
        createArtifactRequest.modelArtifact(Files.newInputStream(zipFile));
        createArtifactRequest.modelId(modelId);
        createArtifactRequest.contentDisposition("attachment; filename=\""+zipFile.toString()+"\"");
        CreateModelArtifactResponse artifactResponse = client.createModelArtifact(createArtifactRequest.build());
        logger.info("Create artifact response: \n"+artifactResponse);

        return modelId;
    }

    /**
     * Creates a Model deployment from an uploaded model.
     * @param config The model deployment configuration
     * @param client The DS Client.
     * @param mapper The object mapper for converting between JSON and objects. Must be configured with the appropriate filters.
     * @throws IOException If the deployment failed.
     * @return The model deployment URL.
     */
    public static String deploy(OCIModelDeploymentConfig config, DataScienceClient client, ObjectMapper mapper) throws IOException {
        // Create the deployment
        CreateModelDeploymentRequest.Builder modelDeploymentRequest = CreateModelDeploymentRequest.builder();
        modelDeploymentRequest.createModelDeploymentDetails(CreateModelDeploymentDetails.builder()
                .projectId(config.dsConfig.projectID)
                .displayName(config.deploymentName)
                .compartmentId(config.dsConfig.compartmentID)
                .modelDeploymentConfigurationDetails(SingleModelDeploymentConfigurationDetails.builder()
                        .modelConfigurationDetails(ModelConfigurationDetails.builder()
                                .modelId(config.modelID)
                                .instanceConfiguration(InstanceConfiguration.builder()
                                        .instanceShapeName(config.shape)
                                        .build())
                                .bandwidthMbps(config.bandwidth).scalingPolicy(FixedSizeScalingPolicy.builder()
                                        .instanceCount(config.instanceCount).build())
                                .build())
                        .build())
                .build());

        ModelDeployment deploymentResponse = client.createModelDeployment(modelDeploymentRequest.build()).getModelDeployment();
        String deploymentResponseStr = mapper.writeValueAsString(deploymentResponse);
        logger.info("Create deployment response: \n"+deploymentResponseStr);
        return deploymentResponse.getModelDeploymentUrl();
    }
}
