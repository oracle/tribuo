/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.onnx;

import ai.onnxruntime.OnnxModelMetadata;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.oracle.labs.mlrg.olcut.config.protobuf.ProtoProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.io.ProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.io.ProvenanceSerializationException;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.interop.ExternalDatasetProvenance;
import org.tribuo.interop.ExternalModel;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.math.la.SparseVector;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A Tribuo wrapper around a ONNX model.
 * <p>
 * N.B. ONNX support is experimental, and may change without a major version bump.
 */
public final class ONNXExternalModel<T extends Output<T>> extends ExternalModel<T, OnnxTensor, List<OnnxValue>> implements AutoCloseable {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ONNXExternalModel.class.getName());

    private transient OrtEnvironment env;

    private transient OrtSession.SessionOptions options;

    private transient OrtSession session;

    private final byte[] modelArray;

    private final String inputName;

    private final ExampleTransformer featureTransformer;

    private final OutputTransformer<T> outputTransformer;

    private ONNXExternalModel(String name, ModelProvenance provenance,
                              ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                              Map<String, Integer> featureMapping,
                              byte[] modelArray, OrtSession.SessionOptions options, String inputName,
                              ExampleTransformer featureTransformer, OutputTransformer<T> outputTransformer) throws OrtException {
        super(name, provenance, featureIDMap, outputIDInfo, outputTransformer.generatesProbabilities(), featureMapping);
        this.modelArray = modelArray;
        this.options = options;
        this.inputName = inputName;
        this.featureTransformer = featureTransformer;
        this.outputTransformer = outputTransformer;
        this.env = OrtEnvironment.getEnvironment("tribuo-" + name);
        this.session = env.createSession(modelArray, options);
    }

    private ONNXExternalModel(String name, ModelProvenance provenance,
                              ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                              int[] featureForwardMapping, int[] featureBackwardMapping,
                              byte[] modelArray, OrtSession.SessionOptions options, String inputName,
                              ExampleTransformer featureTransformer, OutputTransformer<T> outputTransformer) throws OrtException {
        super(name, provenance, featureIDMap, outputIDInfo, featureForwardMapping, featureBackwardMapping,
                outputTransformer.generatesProbabilities());
        this.modelArray = modelArray;
        this.options = options;
        this.inputName = inputName;
        this.featureTransformer = featureTransformer;
        this.outputTransformer = outputTransformer;
        this.env = OrtEnvironment.getEnvironment("tribuo-" + name);
        this.session = env.createSession(modelArray, options);
    }

    /**
     * Closes the session and rebuilds it using the supplied options.
     * <p>
     * Used to select a different backend, or change the number of inference threads etc.
     *
     * @param newOptions The new session options.
     * @throws OrtException If the model failed to rebuild the session with the supplied options.
     */
    public synchronized void rebuild(OrtSession.SessionOptions newOptions) throws OrtException {
        session.close();
        if (options != null) {
            options.close();
        }
        options = newOptions;
        env.createSession(modelArray, newOptions);
    }

    @Override
    protected OnnxTensor convertFeatures(SparseVector input) {
        try {
            return featureTransformer.transform(env, input);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to construct input OnnxTensor", e);
        }
    }

    @Override
    protected OnnxTensor convertFeaturesList(List<SparseVector> input) {
        try {
            return featureTransformer.transform(env, input);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to construct input OnnxTensor", e);
        }
    }

    /**
     * Runs the session to make a prediction.
     * <p>
     * Closes the input tensor after the prediction has been made.
     *
     * @param input The input in the external model's format.
     * @return A tensor representing the output.
     */
    @Override
    protected List<OnnxValue> externalPrediction(OnnxTensor input) {
        try {
            // Note the output of the session is closed by the conversion methods, and should not be closed by the result object.
            OrtSession.Result output = session.run(Collections.singletonMap(inputName, input));
            input.close();
            ArrayList<OnnxValue> outputs = new ArrayList<>();
            for (Map.Entry<String, OnnxValue> v : output) {
                outputs.add(v.getValue());
            }
            return outputs;
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to execute ONNX model", e);
        }
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     *
     * @param output           The output of the external model.
     * @param numValidFeatures The number of valid features in the input.
     * @param example          The input example, used to construct the Prediction.
     * @return A {@link Prediction} representing this tensor output.
     */
    @Override
    protected Prediction<T> convertOutput(List<OnnxValue> output, int numValidFeatures, Example<T> example) {
        Prediction<T> pred = outputTransformer.transformToPrediction(output, outputIDInfo, numValidFeatures, example);
        OnnxValue.close(output);
        return pred;
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     *
     * @param output           The output of the external model.
     * @param numValidFeatures An array with the number of valid features in each example.
     * @param examples         The input examples, used to construct the Predictions.
     * @return A list of {@link Prediction} representing this tensor output.
     */
    @Override
    protected List<Prediction<T>> convertOutput(List<OnnxValue> output, int[] numValidFeatures, List<Example<T>> examples) {
        List<Prediction<T>> predictions = outputTransformer.transformToBatchPrediction(output, outputIDInfo, numValidFeatures, examples);
        OnnxValue.close(output);
        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    protected synchronized Model<T> copy(String newName, ModelProvenance newProvenance) {
        byte[] newModelArray = Arrays.copyOf(modelArray, modelArray.length);
        try {
            return new ONNXExternalModel<>(newName, newProvenance, featureIDMap, outputIDInfo,
                    featureForwardMapping, featureBackwardMapping,
                    newModelArray, options, inputName, featureTransformer, outputTransformer);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to copy ONNX model", e);
        }
    }

    @Override
    public void close() {
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                logger.log(Level.SEVERE, "Exception thrown when closing session", e);
            }
        }
        if (options != null) {
            options.close();
        }
        if (env != null) {
            try {
                env.close();
            } catch (OrtException e) {
                logger.log(Level.SEVERE, "Exception thrown when closing environment", e);
            }
        }
    }

    /**
     * Returns the model provenance from the ONNX model if that
     * model was trained in Tribuo.
     * <p>
     * Tribuo's ONNX export functionality stores the model provenance inside the
     * ONNX file in the metadata field {@link ONNXExportable#PROVENANCE_METADATA_FIELD},
     * and this method provides the access point for it.
     * <p>
     * Note it is different from the {@link Model#getProvenance()} call which
     * returns information about the ONNX file itself, and when the {@code ONNXExternalModel}
     * was created. It does not replace that provenance because instantiating this provenance
     * may require classes which are not present on the classpath at deployment time.
     *
     * @return The model provenance from the original Tribuo training run, if it exists, and
     * returns {@link Optional#empty()} otherwise.
     */
    public Optional<ModelProvenance> getTribuoProvenance() {
        try {
            OnnxModelMetadata metadata = session.getMetadata();
            Optional<String> value = metadata.getCustomMetadataValue(ONNXExportable.PROVENANCE_METADATA_FIELD);
            if (value.isPresent()) {
                Provenance prov = ONNXExportable.SERIALIZER.deserializeAndUnmarshal(value.get());

                if (prov instanceof ModelProvenance) {
                    return Optional.of((ModelProvenance) prov);
                } else {
                    logger.log(Level.WARNING, "Found invalid provenance object, " + prov.toString());
                    return Optional.empty();
                }
            } else {
                return Optional.empty();
            }
        } catch (OrtException e) {
            logger.log(Level.WARNING,"ORTException when reading session metadata",e);
            return Optional.empty();
        } catch (ProvenanceSerializationException e) {
            logger.log(Level.WARNING, "Failed to parse provenance from value.",e);
            return Optional.empty();
        }
    }

    /**
     * Creates an {@code ONNXExternalModel} by loading the model from disk.
     *
     * @param factory            The output factory to use.
     * @param featureMapping     The feature mapping between Tribuo names and ONNX integer ids.
     * @param outputMapping      The output mapping between Tribuo outputs and ONNX integer ids.
     * @param featureTransformer The transformation function for the features.
     * @param outputTransformer  The transformation function for the outputs.
     * @param opts               The session options for the ONNX model.
     * @param filename           The model path.
     * @param inputName          The name of the input node.
     * @param <T>                The type of the output.
     * @return An ONNXExternalModel ready to score new inputs.
     * @throws OrtException If the onnx-runtime native library call failed.
     */
    public static <T extends Output<T>> ONNXExternalModel<T> createOnnxModel(OutputFactory<T> factory,
                                                                             Map<String, Integer> featureMapping,
                                                                             Map<T, Integer> outputMapping,
                                                                             ExampleTransformer featureTransformer,
                                                                             OutputTransformer<T> outputTransformer,
                                                                             OrtSession.SessionOptions opts,
                                                                             String filename, String inputName) throws OrtException {
        Path path = Paths.get(filename);
        return createOnnxModel(factory, featureMapping, outputMapping, featureTransformer, outputTransformer,
                opts, path, inputName);
    }

    /**
     * Creates an {@code ONNXExternalModel} by loading the model from disk.
     *
     * @param factory            The output factory to use.
     * @param featureMapping     The feature mapping between Tribuo names and ONNX integer ids.
     * @param outputMapping      The output mapping between Tribuo outputs and ONNX integer ids.
     * @param featureTransformer The transformation function for the features.
     * @param outputTransformer  The transformation function for the outputs.
     * @param opts               The session options for the ONNX model.
     * @param path               The model path.
     * @param inputName          The name of the input node.
     * @param <T>                The type of the output.
     * @return An ONNXExternalModel ready to score new inputs.
     * @throws OrtException If the onnx-runtime native library call failed.
     */
    public static <T extends Output<T>> ONNXExternalModel<T> createOnnxModel(OutputFactory<T> factory,
                                                                             Map<String, Integer> featureMapping,
                                                                             Map<T, Integer> outputMapping,
                                                                             ExampleTransformer featureTransformer,
                                                                             OutputTransformer<T> outputTransformer,
                                                                             OrtSession.SessionOptions opts,
                                                                             Path path, String inputName) throws OrtException {
        try {
            byte[] modelArray = Files.readAllBytes(path);
            URL provenanceLocation = path.toUri().toURL();
            ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
            ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory, outputMapping);
            OffsetDateTime now = OffsetDateTime.now();
            ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance(provenanceLocation);
            DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data", factory, false, featureMapping.size(), outputMapping.size());
            HashMap<String, Provenance> runProvenance = new HashMap<>();
            runProvenance.put("input-name", new StringProvenance("input-name", inputName));
            try (OrtEnvironment env = OrtEnvironment.getEnvironment();
                 OrtSession session = env.createSession(modelArray)) {
                OnnxModelMetadata metadata = session.getMetadata();
                runProvenance.put("model-producer", new StringProvenance("model-producer", metadata.getProducerName()));
                runProvenance.put("model-domain", new StringProvenance("model-domain", metadata.getDomain()));
                runProvenance.put("model-description", new StringProvenance("model-description", metadata.getDescription()));
                runProvenance.put("model-graphname", new StringProvenance("model-graphname", metadata.getGraphName()));
                runProvenance.put("model-version", new LongProvenance("model-version", metadata.getVersion()));
                for (Map.Entry<String, String> e : metadata.getCustomMetadata().entrySet()) {
                    String keyName = "model-metadata-" + e.getKey();
                    runProvenance.put(keyName, new StringProvenance(keyName, e.getValue()));
                }
            } catch (OrtException e) {
                throw new IllegalArgumentException("Failed to load model and read metadata from path " + path, e);
            }
            ModelProvenance provenance = new ModelProvenance(ONNXExternalModel.class.getName(), now, datasetProvenance, trainerProvenance, runProvenance);
            return new ONNXExternalModel<>("external-model", provenance, featureMap, outputInfo,
                    featureMapping, modelArray, opts, inputName, featureTransformer, outputTransformer);
        } catch (IOException e) {
            throw new IllegalArgumentException("Unable to load model from path " + path, e);
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        try {
            this.env = OrtEnvironment.getEnvironment();
            this.options = new OrtSession.SessionOptions();
            this.session = env.createSession(modelArray, options);
        } catch (OrtException e) {
            throw new IllegalStateException("Could not construct ONNX Runtime session during deserialization.");
        }
    }

}
