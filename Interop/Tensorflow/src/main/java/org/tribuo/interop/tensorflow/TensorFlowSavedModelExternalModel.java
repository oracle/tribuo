/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.tensorflow;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.exceptions.TensorFlowException;
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
import org.tribuo.interop.tensorflow.protos.TensorFlowSavedModelExternalModelProto;
import org.tribuo.math.la.SparseVector;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.Util;

import java.io.Closeable;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * A Tribuo wrapper around a TensorFlow saved model bundle.
 * <p>
 * The model's serialVersionUID is set to the major TensorFlow version number times 100.
 * <p>
 * N.B. TensorFlow support is experimental and may change without a major version bump.
 */
public final class TensorFlowSavedModelExternalModel<T extends Output<T>> extends ExternalModel<T, TensorMap, TensorMap> implements Closeable {
    private static final long serialVersionUID = 200L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final String modelDirectory;

    private transient SavedModelBundle bundle;

    private final FeatureConverter featureConverter;

    private final OutputConverter<T> outputConverter;

    private final String outputName;

    private TensorFlowSavedModelExternalModel(String name, ModelProvenance provenance,
                                              ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                              Map<String, Integer> featureMapping,
                                              String modelDirectory, String outputName,
                                              FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name, provenance, featureIDMap, outputIDInfo, outputConverter.generatesProbabilities(), featureMapping);
        this.modelDirectory = modelDirectory;
        this.outputName = outputName;
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
        SavedModelBundle.Loader loader = SavedModelBundle.loader(modelDirectory);
        bundle = loader.load();
    }

    private TensorFlowSavedModelExternalModel(String name, ModelProvenance provenance,
                                              ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                              int[] featureForwardMapping, int[] featureBackwardMapping,
                                              String modelDirectory, String outputName,
                                              FeatureConverter featureConverter, OutputConverter<T> outputConverter) {
        super(name,provenance,featureIDMap,outputIDInfo,featureForwardMapping,featureBackwardMapping,
                outputConverter.generatesProbabilities());
        this.modelDirectory = modelDirectory;
        this.outputName = outputName;
        this.featureConverter = featureConverter;
        this.outputConverter = outputConverter;
        SavedModelBundle.Loader loader = SavedModelBundle.loader(modelDirectory);
        bundle = loader.load();
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"rawtypes","unchecked"}) // guarded by a getClass check that the output domain and converter are compatible
    public static TensorFlowSavedModelExternalModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        TensorFlowSavedModelExternalModelProto proto = message.unpack(TensorFlowSavedModelExternalModelProto.class);

        OutputConverter<?> outputConverter = ProtoUtil.deserialize(proto.getOutputConverter());
        FeatureConverter featureConverter = ProtoUtil.deserialize(proto.getFeatureConverter());

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(outputConverter.getTypeWitness())) {
            throw new IllegalStateException("Invalid protobuf, output domain does not match converter, found " + carrier.outputDomain().getClass() + " and " + outputConverter.getTypeWitness());
        }
        int[] featureForwardMapping = Util.toPrimitiveInt(proto.getForwardFeatureMappingList());
        int[] featureBackwardMapping = Util.toPrimitiveInt(proto.getBackwardFeatureMappingList());
        if (!validateFeatureMapping(featureForwardMapping,featureBackwardMapping,carrier.featureDomain())) {
            throw new IllegalStateException("Invalid protobuf, external<->Tribuo feature mapping does not form a bijection");
        }

        return new TensorFlowSavedModelExternalModel(carrier.name(), carrier.provenance(), carrier.featureDomain(),
                carrier.outputDomain(), featureForwardMapping, featureBackwardMapping, proto.getModelDirectory(),
                proto.getOutputName(), featureConverter, outputConverter);
    }

    @Override
    protected TensorMap convertFeatures(SparseVector input) {
        return featureConverter.convert(input);
    }

    @Override
    protected TensorMap convertFeaturesList(List<SparseVector> input) {
        return featureConverter.convert(input);
    }

    /**
     * Runs the session to make a prediction.
     * <p>
     * Closes the input tensor after the prediction has been made.
     * @param input The input in the external model's format.
     * @return A tensor representing the output.
     */
    @Override
    protected TensorMap externalPrediction(TensorMap input) {
        Map<String,Tensor> output = bundle.call(input.getMap());
        input.close();
        return new TensorMap(output);
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     * @param output The output of the external model.
     * @param numValidFeatures The number of valid features in the input.
     * @param example The input example, used to construct the Prediction.
     * @return A {@link Prediction} representing this tensor output.
     */
    @Override
    protected Prediction<T> convertOutput(TensorMap output, int numValidFeatures, Example<T> example) {
        Optional<Tensor> tensor = output.getTensor(outputName);
        if (tensor.isPresent()) {
            Prediction<T> pred = outputConverter.convertToPrediction(tensor.get(), outputIDInfo, numValidFeatures, example);
            output.close();
            return pred;
        } else {
            output.close();
            throw new IllegalArgumentException("Failed to find '" + outputName + "' in model output. Found " + output);
        }
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     * @param output The output of the external model.
     * @param numValidFeatures An array with the number of valid features in each example.
     * @param examples The input examples, used to construct the Predictions.
     * @return A list of {@link Prediction} representing this tensor output.
     */
    @Override
    protected List<Prediction<T>> convertOutput(TensorMap output, int[] numValidFeatures, List<Example<T>> examples) {
        Optional<Tensor> tensor = output.getTensor(outputName);
        if (tensor.isPresent()) {
            List<Prediction<T>> predictions = outputConverter.convertToBatchPrediction(tensor.get(),outputIDInfo,numValidFeatures,examples);
            output.close();
            return predictions;
        } else {
            output.close();
            throw new IllegalArgumentException("Failed to find '" + outputName + "' in model output. Found " + output);
        }
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    protected Model<T> copy(String newName, ModelProvenance newProvenance) {
        return new TensorFlowSavedModelExternalModel<>(newName,newProvenance,featureIDMap,outputIDInfo,
                featureForwardMapping,featureBackwardMapping,
                modelDirectory,outputName,featureConverter, outputConverter);
    }

    @Override
    public void close() {
        if (bundle != null) {
            bundle.close();
        }
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        TensorFlowSavedModelExternalModelProto.Builder modelBuilder = TensorFlowSavedModelExternalModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setOutputName(outputName);
        modelBuilder.setModelDirectory(modelDirectory);
        modelBuilder.addAllForwardFeatureMapping(Arrays.stream(featureForwardMapping).boxed().collect(Collectors.toList()));
        modelBuilder.addAllBackwardFeatureMapping(Arrays.stream(featureBackwardMapping).boxed().collect(Collectors.toList()));
        modelBuilder.setOutputConverter(outputConverter.serialize());
        modelBuilder.setFeatureConverter(featureConverter.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(TensorFlowSavedModelExternalModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    /**
     * Creates a TensorflowSavedModelExternalModel by loading in a {@code SavedModelBundle}.
     * <p>
     * Throws {@link IllegalArgumentException} if the model bundle could not be loaded.
     * @param factory The output factory.
     * @param featureMapping The feature mapping between Tribuo's names and the TF integer ids.
     * @param outputMapping The output mapping between Tribuo's names and the TF integer ids.
     * @param outputName The name of the output tensor.
     * @param featureConverter The feature transformation function.
     * @param outputConverter The output transformation function.
     * @param bundleDirectory The path to load the saved model bundle from.
     * @param <T> The type of the output.
     * @return The TF model wrapped in a Tribuo {@link ExternalModel}.
     */
    public static <T extends Output<T>> TensorFlowSavedModelExternalModel<T> createTensorflowModel(OutputFactory<T> factory,
                                                                                                   Map<String, Integer> featureMapping,
                                                                                                   Map<T,Integer> outputMapping,
                                                                                                   String outputName,
                                                                                                   FeatureConverter featureConverter,
                                                                                                   OutputConverter<T> outputConverter,
                                                                                                   String bundleDirectory) {
        try {
            Path path = Paths.get(bundleDirectory);
            URL provenanceLocation = path.toUri().toURL();
            ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
            ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory, outputMapping);
            OffsetDateTime now = OffsetDateTime.now();
            ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance(provenanceLocation);
            DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data", factory, false, featureMapping.size(), outputMapping.size());
            ModelProvenance provenance = new ModelProvenance(TensorFlowSavedModelExternalModel.class.getName(), now, datasetProvenance, trainerProvenance);
            return new TensorFlowSavedModelExternalModel<>("tf-saved-model-bundle", provenance, featureMap, outputInfo,
                    featureMapping, bundleDirectory, outputName, featureConverter, outputConverter);
        } catch (IOException | TensorFlowException e) {
            throw new IllegalArgumentException("Unable to load model from path " + bundleDirectory, e);
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        SavedModelBundle.Loader loader = SavedModelBundle.loader(modelDirectory);
        bundle = loader.load();
    }

}
