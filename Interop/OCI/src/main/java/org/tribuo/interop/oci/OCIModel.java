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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.bmc.ConfigFileReader;
import com.oracle.bmc.auth.AuthenticationDetailsProvider;
import com.oracle.bmc.auth.ConfigFileAuthenticationDetailsProvider;
import com.oracle.bmc.http.signing.RequestSigningFilter;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.FileProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.interop.ExternalDatasetProvenance;
import org.tribuo.interop.ExternalModel;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.interop.oci.protos.OCIModelProto;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
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
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * A wrapper class around an OCI Data Science Model Deployment endpoint which sends off inputs for scoring and
 * converts the output into a Tribuo prediction.
 */
public final class OCIModel<T extends Output<T>> extends ExternalModel<T, DenseMatrix, DenseMatrix> implements AutoCloseable {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(OCIModel.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final Path configFile;
    private final String profileName;
    private final String endpointURL;
    private final String modelDeploymentId;
    private final OCIOutputConverter<T> outputConverter;

    // Derived state
    private transient AuthenticationDetailsProvider authProvider;
    private transient RequestSigningFilter requestSigningFilter;
    private transient Client jerseyClient;
    private transient WebTarget modelEndpoint;
    private transient ObjectMapper mapper;

    /**
     * Construct an OCIModel wrapping an OCI DS Model Deployment endpoint.
     *
     * @param name              The model name.
     * @param provenance        The model provenance.
     * @param featureIDMap      The feature map.
     * @param outputIDInfo      The output map.
     * @param featureMapping    The mapping between Tribuo's feature names and the external model's feature indices.
     * @param configFile        The OCI configuration file, if null use the default file.
     * @param profileName       The OCI client profile, or null for the default.
     * @param endpointURL       The OCI Model Deployment endpoint URL.
     * @param modelDeploymentId The model deployment ID.
     * @param outputConverter   The output conversion function.
     * @throws IOException If the OCI configuration file could not be read.
     */
    OCIModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
             Map<String, Integer> featureMapping,
             Path configFile, String profileName, String endpointURL, String modelDeploymentId, OCIOutputConverter<T> outputConverter) throws IOException {
        super(name, provenance, featureIDMap, outputIDInfo, outputConverter.generatesProbabilities(), featureMapping);
        this.configFile = configFile;
        this.profileName = profileName;
        this.endpointURL = endpointURL;
        this.modelDeploymentId = modelDeploymentId;
        this.outputConverter = outputConverter;
        this.authProvider = makeAuthProvider(configFile,profileName);
        this.mapper = new ObjectMapper();

        // Pre-Requirement: Allow setting of restricted headers. This is required to allow the SigningFilter
        // to set the host header that gets computed during signing of the request.
        System.setProperty("sun.net.http.allowRestrictedHeaders", "true");

        // 1) Create a request signing filter instance
        this.requestSigningFilter = RequestSigningFilter.fromAuthProvider(authProvider);

        // 2) Create a Jersey client and register the request signing filter
        this.jerseyClient = ClientBuilder.newBuilder().build().register(requestSigningFilter);

        // 3) Target an endpoint.
        this.modelEndpoint = jerseyClient.target(endpointURL + modelDeploymentId).path("predict");
    }

    /**
     * Construct an OCIModel wrapping an OCI DS Model Deployment endpoint.
     *
     * @param name                   The model name.
     * @param provenance             The model provenance.
     * @param featureIDMap           The feature map.
     * @param outputIDInfo           The output map.
     * @param featureForwardMapping  The forward mapping between Tribuo's feature indices and the external ones.
     * @param featureBackwardMapping The backward mapping between Tribuo's feature indices and the external ones.
     * @param configFile             The OCI configuration file, if null use the default file.
     * @param authProvider           The OCI authentication provider.
     * @param endpointURL            The OCI Model Deployment endpoint URL.
     * @param modelDeploymentId      The model deployment ID.
     * @param outputConverter        The output conversion function.
     */
    OCIModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
             int[] featureForwardMapping, int[] featureBackwardMapping,
             Path configFile, String profileName,
             AuthenticationDetailsProvider authProvider, String endpointURL, String modelDeploymentId, OCIOutputConverter<T> outputConverter) {
        super(name, provenance, featureIDMap, outputIDInfo, featureForwardMapping, featureBackwardMapping, outputConverter.generatesProbabilities());
        this.configFile = configFile;
        this.profileName = profileName;
        this.authProvider = authProvider;
        this.endpointURL = endpointURL;
        this.modelDeploymentId = modelDeploymentId;
        this.outputConverter = outputConverter;
        this.mapper = new ObjectMapper();

        // Pre-Requirement: Allow setting of restricted headers. This is required to allow the SigningFilter
        // to set the host header that gets computed during signing of the request.
        System.setProperty("sun.net.http.allowRestrictedHeaders", "true");

        // 1) Create a request signing filter instance
        this.requestSigningFilter = RequestSigningFilter.fromAuthProvider(authProvider);

        // 2) Create a Jersey client and register the request signing filter
        this.jerseyClient = ClientBuilder.newBuilder().build().register(requestSigningFilter);

        // 3) Target an endpoint.
        this.modelEndpoint = jerseyClient.target(endpointURL + modelDeploymentId).path("predict");
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"rawtypes","unchecked"}) // guarded by a getClass check
    public static OCIModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException, IOException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        OCIModelProto proto = message.unpack(OCIModelProto.class);
        OCIOutputConverter<?> converter = ProtoUtil.deserialize(proto.getOutputConverter());

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(converter.getTypeWitness())) {
            throw new IllegalStateException("Invalid protobuf, output domain does not match converter, found " + carrier.outputDomain().getClass() + " and " + converter.getTypeWitness());
        }
        Path configFile = Paths.get(proto.getConfigFile());
        AuthenticationDetailsProvider authProvider = makeAuthProvider(configFile, proto.getProfileName());
        int[] featureForwardMapping = Util.toPrimitiveInt(proto.getForwardFeatureMappingList());
        int[] featureBackwardMapping = Util.toPrimitiveInt(proto.getBackwardFeatureMappingList());
        if (!validateFeatureMapping(featureForwardMapping,featureBackwardMapping,carrier.featureDomain())) {
            throw new IllegalStateException("Invalid protobuf, external<->Tribuo feature mapping does not form a bijection");
        }

        return new OCIModel(carrier.name(), carrier.provenance(), carrier.featureDomain(), carrier.outputDomain(),
                featureForwardMapping, featureBackwardMapping, configFile, proto.getProfileName(), authProvider,
                proto.getEndpointUrl(), proto.getModelDeploymentId(), converter);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int i) {
        return Collections.emptyMap();
    }

    @Override
    protected Model<T> copy(String s, ModelProvenance modelProvenance) {
        return new OCIModel<>(s, modelProvenance, featureIDMap, outputIDInfo, featureForwardMapping, featureBackwardMapping, configFile, profileName, authProvider, endpointURL, modelDeploymentId, outputConverter);
    }

    @Override
    protected DenseMatrix convertFeatures(SparseVector sparseVector) {
        return DenseMatrix.createDenseMatrix(new double[][]{sparseVector.toArray()});
    }

    @Override
    protected DenseMatrix convertFeaturesList(List<SparseVector> list) {
        double[][] inputs = new double[list.size()][];
        for (int i = 0; i < list.size(); i++) {
            SparseVector v = list.get(i);
            inputs[i] = v.toArray();
        }
        return DenseMatrix.createDenseMatrix(inputs);
    }

    @Override
    protected DenseMatrix externalPrediction(DenseMatrix features) {
        Invocation.Builder ib = modelEndpoint.request();
        ib.accept(MediaType.APPLICATION_JSON);
        Response response = ib.buildPost(Entity.entity(formatMatrix(features), MediaType.APPLICATION_JSON)).invoke();

        String json;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader((InputStream) response.getEntity(), StandardCharsets.UTF_8))) {
            StringBuilder jsonBody = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                jsonBody.append(line);
            }
            json = jsonBody.toString();
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read response from input stream", e);
        }
        try {
            PredictionJson predJson = mapper.readValue(json, OCIModel.PredictionJson.class);
            return DenseMatrix.createDenseMatrix(predJson.prediction);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Failed to parse json from deployed model endpoint, received '" + json + "'", e);
        }
    }

    /**
     * Formats a Tribuo DenseMatrix as a nested Json array.
     *
     * @param matrix The matrix to format.
     * @return A Json array string.
     */
    private static String formatMatrix(DenseMatrix matrix) {
        StringBuilder builder = new StringBuilder();
        builder.append('[');
        for (int i = 0; i < matrix.getDimension1Size(); i++) {
            builder.append(Arrays.toString(matrix.getRow(i).toArray()));
            builder.append(',');
        }
        builder.setCharAt(builder.length() - 1, ']');
        return builder.toString();
    }

    @Override
    protected Prediction<T> convertOutput(DenseMatrix scores, int numValidFeature, Example<T> example) {
        if (scores.getDimension1Size() != 1) {
            throw new IllegalStateException("Expected a single score vector, received " + scores.getDimension1Size());
        }
        return outputConverter.convertOutput(scores.getRow(0), numValidFeature, example, outputIDInfo);
    }

    @Override
    protected List<Prediction<T>> convertOutput(DenseMatrix scores, int[] numValidFeatures, List<Example<T>> list) {
        return outputConverter.convertOutput(scores, numValidFeatures, list, outputIDInfo);
    }

    @Override
    public void close() {
        jerseyClient.close();
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        OCIModelProto.Builder modelBuilder = OCIModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.addAllForwardFeatureMapping(Arrays.stream(featureForwardMapping).boxed().collect(Collectors.toList()));
        modelBuilder.addAllBackwardFeatureMapping(Arrays.stream(featureBackwardMapping).boxed().collect(Collectors.toList()));
        modelBuilder.setConfigFile(configFile.toString());
        modelBuilder.setProfileName(profileName);
        modelBuilder.setEndpointUrl(endpointURL);
        modelBuilder.setModelDeploymentId(modelDeploymentId);
        modelBuilder.setOutputConverter(outputConverter.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(OCIModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        // Pre-Requirement: Allow setting of restricted headers. This is required to allow the SigningFilter
        // to set the host header that gets computed during signing of the request.
        System.setProperty("sun.net.http.allowRestrictedHeaders", "true");

        // Rebuild transient state
        this.authProvider = makeAuthProvider(configFile,profileName);
        this.mapper = new ObjectMapper();
        this.requestSigningFilter = RequestSigningFilter.fromAuthProvider(authProvider);
        this.jerseyClient = ClientBuilder.newBuilder().build().register(requestSigningFilter);
        this.modelEndpoint = jerseyClient.target(endpointURL + modelDeploymentId).path("predict");
    }

    /**
     * Carrier type for easy deserialization from JSON.
     */
    public static final class PredictionJson {
        /**
         * The predicted probabilities or scores.
         */
        @JsonProperty("prediction")
        public double[][] prediction;

        /**
         * Constructs a prediction object.
         * @param prediction The predicted probabilities or scores.
         */
        @JsonCreator
        public PredictionJson(@JsonProperty("prediction") double[][] prediction) {
            this.prediction = prediction;
        }
    }

    /**
     * Makes an authentication provider from the config file and profile.
     * <p>
     * If the config file is null then it loads the default config file, if the profile is null it loads the
     * default profile in the chosen config file.
     * @param configFile The config file to load.
     * @param profile The profile to load.
     * @return An authentication provider.
     * @throws IOException If the config file could not be read.
     */
    public static ConfigFileAuthenticationDetailsProvider makeAuthProvider(Path configFile, String profile) throws IOException {
        ConfigFileReader.ConfigFile file = configFile == null ? ConfigFileReader.parseDefault(profile) : ConfigFileReader.parse(configFile.toString());
        return new ConfigFileAuthenticationDetailsProvider(file);
    }

    /**
     * Creates an {@code OCIModel} by wrapping an OCI DS Model Deployment endpoint.
     * <p>
     * Uses the endpointURL as the value to hash for the trainer provenance.
     * <p>
     * Loads the default profile in the configuration file.
     *
     * @param factory         The output factory to use.
     * @param featureMapping  The feature mapping between Tribuo names and model integer ids.
     * @param outputMapping   The output mapping between Tribuo outputs and model integer ids.
     * @param configFile      The OCI configuration file, if null use the default file.
     * @param endpointURL     The endpoint URL.
     * @param outputConverter The converter for the specified output type.
     * @param <T> The output type.
     * @return An OCIModel ready to score new inputs.
     */
    public static <T extends Output<T>> OCIModel<T> createOCIModel(OutputFactory<T> factory,
                                                                   Map<String, Integer> featureMapping,
                                                                   Map<T, Integer> outputMapping,
                                                                   Path configFile,
                                                                   String endpointURL,
                                                                   OCIOutputConverter<T> outputConverter) {
        return createOCIModel(factory, featureMapping, outputMapping, configFile, null, endpointURL, outputConverter);
    }

    /**
     * Creates an {@code OCIModel} by wrapping an OCI DS Model Deployment endpoint.
     * <p>
     * Uses the endpointURL as the value to hash for the trainer provenance.
     *
     * @param factory         The output factory to use.
     * @param featureMapping  The feature mapping between Tribuo names and model integer ids.
     * @param outputMapping   The output mapping between Tribuo outputs and model integer ids.
     * @param configFile      The OCI configuration file, if null use the default file.
     * @param profileName     The profile name in the OCI configuration file, if null uses the default profile.
     * @param endpointURL     The endpoint URL.
     * @param outputConverter The converter for the specified output type.
     * @param <T> The output type.
     * @return An OCIModel ready to score new inputs.
     */
    public static <T extends Output<T>> OCIModel<T> createOCIModel(OutputFactory<T> factory,
                                                                   Map<String, Integer> featureMapping,
                                                                   Map<T, Integer> outputMapping,
                                                                   Path configFile,
                                                                   String profileName,
                                                                   String endpointURL,
                                                                   OCIOutputConverter<T> outputConverter) {
        try {
            ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
            ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory, outputMapping);
            OffsetDateTime now = OffsetDateTime.now();
            ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance((endpointURL).getBytes(StandardCharsets.UTF_8));
            DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data", factory, false, featureMapping.size(), outputMapping.size());
            String[] endpoint = endpointURL.split("/");
            String domain = "https://" + endpoint[2] + "/";
            String modelDeploymentId = endpoint[3];
            HashMap<String, Provenance> runProvenance = new HashMap<>();
            runProvenance.put("configFile", new FileProvenance("configFile", configFile));
            runProvenance.put("endpointURL", new StringProvenance("endpointURL", endpointURL));
            runProvenance.put("modelDeploymentId", new StringProvenance("modelDeploymentId", modelDeploymentId));
            ModelProvenance provenance = new ModelProvenance(OCIModel.class.getName(), now, datasetProvenance, trainerProvenance, runProvenance);
            return new OCIModel<T>("oci-ds-model", provenance, featureMap, outputInfo, featureMapping, configFile,
                    profileName, domain, modelDeploymentId, outputConverter);
        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to load configuration from path " + configFile, e);
        }
    }

}
