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

package org.tribuo.ensemble;

import ai.onnx.proto.OnnxMl;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.ONNXExportable;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.WeightedEnsembleModelProto;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.impl.TimestampedTrainerProvenance;
import org.tribuo.util.Util;
import org.tribuo.util.onnx.ONNXContext;
import org.tribuo.util.onnx.ONNXInitializer;
import org.tribuo.util.onnx.ONNXNode;
import org.tribuo.util.onnx.ONNXOperators;
import org.tribuo.util.onnx.ONNXPlaceholder;
import org.tribuo.util.onnx.ONNXRef;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;

/**
 * An ensemble model that uses weights to combine the ensemble member predictions.
 */
public final class WeightedEnsembleModel<T extends Output<T>> extends EnsembleModel<T> implements ONNXExportable {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * The ensemble member combination weights.
     */
    protected final float[] weights;

    /**
     * The ensemble combination function.
     */
    protected final EnsembleCombiner<T> combiner;

    /**
     * Unless you are implementing a {@link org.tribuo.Trainer} you should
     * not use this constructor directly. Instead use {@link #createEnsembleFromExistingModels(String, List, EnsembleCombiner)}.
     * <p>
     * Constructs an ensemble model which uses uniform weights.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param newModels The list of ensemble members.
     * @param combiner The combination function.
     */
    public WeightedEnsembleModel(String name, EnsembleModelProvenance provenance, ImmutableFeatureMap featureIDMap,
                                 ImmutableOutputInfo<T> outputIDInfo,
                                 List<Model<T>> newModels, EnsembleCombiner<T> combiner) {
        this(name,provenance,featureIDMap,outputIDInfo,newModels, combiner, Util.generateUniformVector(newModels.size(), 1.0f/newModels.size()));
    }

    /**
     * Unless you are implementing a {@link org.tribuo.Trainer} you should
     * not use this constructor directly. Instead use {@link #createEnsembleFromExistingModels(String, List, EnsembleCombiner, float[])}.
     * <p>
     * Constructs an ensemble model which uses uniform weights.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param newModels The list of ensemble members.
     * @param combiner The combination function.
     * @param weights The model combination weights.
     */
    public WeightedEnsembleModel(String name, EnsembleModelProvenance provenance, ImmutableFeatureMap featureIDMap,
                                 ImmutableOutputInfo<T> outputIDInfo, List<Model<T>> newModels,
                                 EnsembleCombiner<T> combiner, float[] weights) {
        super(name,provenance,featureIDMap,outputIDInfo,newModels);
        this.weights = Arrays.copyOf(weights,weights.length);
        this.combiner = combiner;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    @SuppressWarnings({"unchecked","rawtypes"}) // Guarded by getClass checks to ensure all outputs are the same type.
    public static WeightedEnsembleModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        WeightedEnsembleModelProto proto = message.unpack(WeightedEnsembleModelProto.class);

        ModelDataCarrier<? extends Output<?>> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        ModelProvenance prov = carrier.provenance();
        if (!(prov instanceof EnsembleModelProvenance)) {
            throw new IllegalStateException("Invalid protobuf, the provenance was not an EnsembleModelProvenance. Found " + prov);
        }
        EnsembleModelProvenance ensembleProvenance = (EnsembleModelProvenance) prov;
        ImmutableOutputInfo<? extends Output<?>> outputDomain = carrier.outputDomain();
        Class<? extends Output> outputClass = outputDomain.getOutput(0).getClass();
        EnsembleCombiner<?> combiner = EnsembleCombiner.deserialize(proto.getCombiner());
        if (!outputClass.equals(combiner.getTypeWitness())) {
            throw new IllegalStateException("Invalid protobuf, combiner and output domain have a type mismatch, expected " + outputClass + " found " + combiner.getTypeWitness());
        }

        if (proto.getModelsCount() == 0) {
            throw new IllegalStateException("Invalid protobuf, no models were found in the ensemble");
        }
        if (proto.getModelsCount() != proto.getWeightsCount()) {
            throw new IllegalStateException("Invalid protobuf, different numbers of models and weights were found, " + proto.getModelsCount() + " models, " + proto.getWeightsCount() + " weights");
        }
        List<Model> models = new ArrayList<>(proto.getModelsCount());
        for (ModelProto p : proto.getModelsList()) {
            Model model = Model.deserialize(p);
            if (model.validate(outputClass)) {
                models.add(model);
            } else {
                throw new IllegalStateException("Invalid protobuf, output type of model '" + model.toString() + "' did not match expected " + outputClass);
            }
        }

        float[] weights = Util.toPrimitiveFloat(proto.getWeightsList());

        return new WeightedEnsembleModel(carrier.name(), ensembleProvenance, carrier.featureDomain(), outputDomain,
            models, combiner, weights);
    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        List<Prediction<T>> predictions = new ArrayList<>();
        for (Model<T> model : models) {
            predictions.add(model.predict(example));
        }

        return combiner.combine(outputIDInfo,predictions,weights);
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        Map<String, Map<String,Double>> map = new HashMap<>();
        Prediction<T> prediction = predict(example);
        List<Excuse<T>> excuses = new ArrayList<>();

        for (int i = 0; i < models.size(); i++) {
            Optional<Excuse<T>> excuse = models.get(i).getExcuse(example);
            if (excuse.isPresent()) {
                excuses.add(excuse.get());
                Map<String, List<Pair<String,Double>>> m = excuse.get().getScores();
                for (Entry<String, List<Pair<String,Double>>> e : m.entrySet()) {
                    Map<String, Double> innerMap = map.computeIfAbsent(e.getKey(), k -> new HashMap<>());
                    for (Pair<String,Double> p : e.getValue()) {
                        innerMap.merge(p.getA(), p.getB() * weights[i], Double::sum);
                    }
                }
            }
        }

        if (map.isEmpty()) {
            return Optional.empty();
        } else {
            Map<String, List<Pair<String, Double>>> outputMap = new HashMap<>();
            for (Entry<String, Map<String, Double>> label : map.entrySet()) {
                List<Pair<String, Double>> list = new ArrayList<>();

                for (Entry<String, Double> entry : label.getValue().entrySet()) {
                    list.add(new Pair<>(entry.getKey(), entry.getValue()));
                }

                list.sort((Pair<String, Double> o1, Pair<String, Double> o2) -> o2.getB().compareTo(o1.getB()));
                outputMap.put(label.getKey(), list);
            }

            return Optional.of(new EnsembleExcuse<>(example, prediction, outputMap, excuses));
        }
    }

    @Override
    protected EnsembleModel<T> copy(String name, EnsembleModelProvenance newProvenance, List<Model<T>> newModels) {
        return new WeightedEnsembleModel<>(name,newProvenance,featureIDMap,outputIDInfo,newModels,combiner);
    }

    /**
     * Creates an ensemble from existing models. The model outputs are combined using uniform weights.
     * <p>
     * Uses the feature and output domain from the first model as the ensemble model's domains.
     * The individual ensemble members use the domains that they contain.
     * <p>
     * If the output domains don't cover the same dimensions then it throws {@link IllegalArgumentException}.
     * @param name The ensemble name.
     * @param models The ensemble members.
     * @param combiner The combination function.
     * @param <T> The output type.
     * @return A weighted ensemble model.
     */
    public static <T extends Output<T>> WeightedEnsembleModel<T> createEnsembleFromExistingModels(String name, List<Model<T>> models, EnsembleCombiner<T> combiner) {
        return createEnsembleFromExistingModels(name,models,combiner,Util.generateUniformVector(models.size(), 1.0f/models.size()));
    }

    /**
     * Creates an ensemble from existing models.
     * <p>
     * Uses the feature and output domain from the first model as the ensemble model's domains.
     * The individual ensemble members use the domains that they contain.
     * <p>
     * If the output domains don't cover the same dimensions then it throws {@link IllegalArgumentException}.
     * If the weights aren't the same length as the models it throws {@link IllegalArgumentException}.
     * @param name The ensemble name.
     * @param models The ensemble members.
     * @param combiner The combination function.
     * @param weights The model combination weights.
     * @param <T> The output type.
     * @return A weighted ensemble model.
     */
    public static <T extends Output<T>> WeightedEnsembleModel<T> createEnsembleFromExistingModels(String name, List<Model<T>> models, EnsembleCombiner<T> combiner, float[] weights) {
        // Basic parameter validation
        if (models.size() < 2) {
            throw new IllegalArgumentException("Must supply at least 2 models, found " + models.size());
        }
        if (weights.length != models.size()) {
            throw new IllegalArgumentException("Must supply one weight per model, models.size() = " + models.size() + ", weights.length = " + weights.length);
        }

        // Validate output & feature domains
        ImmutableOutputInfo<T> outputInfo = models.get(0).getOutputIDInfo();
        ImmutableFeatureMap featureMap = models.get(0).getFeatureIDMap();
        Set<T> firstOutputDomain = outputInfo.getDomain();
        for (int i = 1; i < models.size(); i++) {
            if (!models.get(i).getOutputIDInfo().getDomain().equals(firstOutputDomain)) {
                throw new IllegalArgumentException("Model output domains are not equal.");
            }
            if (!models.get(i).getFeatureIDMap().domainEquals(featureMap)) {
                throw new IllegalArgumentException("Model feature domains are not equal.");
            }
        }

        // Defensive copy the model list (the weights are copied in the constructor)
        List<Model<T>> modelList = new ArrayList<>(models);

        // Build EnsembleModelProvenance
        TimestampedTrainerProvenance trainerProvenance = new TimestampedTrainerProvenance();
        EnsembleModelProvenance provenance = new EnsembleModelProvenance(
                WeightedEnsembleModel.class.getName(), OffsetDateTime.now(),
                models.get(0).getProvenance().getDatasetProvenance(),
                trainerProvenance,
                ListProvenance.createListProvenance(models)
                );

        return new WeightedEnsembleModel<>(name,provenance,featureMap,outputInfo,modelList,combiner,weights);
    }

    /**
     * Exports this {@link EnsembleModel} as an ONNX model.
     * <p>
     * Note if the ensemble members or the ensemble combination function do not implement
     * {@link ONNXExportable} then this method will throw {@link UnsupportedOperationException}.
     * @param domain A reverse-DNS name to namespace the model (e.g., org.tribuo.classification.sgd.linear).
     * @param modelVersion A version number for this model.
     * @return An ONNX ModelProto representing the model.
     */
    @Override
    public OnnxMl.ModelProto exportONNXModel(String domain, long modelVersion) {
        ONNXContext onnx = new ONNXContext();

        onnx.setName("WeightedEnsembleModel");

        ONNXPlaceholder input = onnx.floatInput(featureIDMap.size());
        ONNXPlaceholder output = onnx.floatOutput(outputIDInfo.size());

        // Build graph
        writeONNXGraph(input).assignTo(output);

        return ONNXExportable.buildModel(onnx, domain, modelVersion, this);
    }

    @Override
    public ONNXNode writeONNXGraph(ONNXRef<?> input) {
        ONNXContext onnx = input.onnxContext();
        ONNXInitializer unsqueezeAxes = onnx.array("unsqueeze_ensemble_output", new long[]{2});
        List<ONNXNode> unsqueezedMembers = new ArrayList<>();
        for(Model<T> model : models) {
            if(model instanceof ONNXExportable) {
                ONNXNode memberOutput = ((ONNXExportable) model).writeONNXGraph(input);
                ONNXNode unsqueezedOutput = memberOutput.apply(ONNXOperators.UNSQUEEZE, unsqueezeAxes);
                if (model.getOutputIDInfo().domainAndIDEquals(outputIDInfo)) {
                    // Output domains line up
                    unsqueezedMembers.add(unsqueezedOutput);
                } else {
                    // Output domains don't line up, add a gather to rearrange them
                    int[] outputRemapping = new int[outputIDInfo.size()];
                    for (int i = 0; i < outputRemapping.length; i++) {
                        int otherId = outputIDInfo.getID(model.getOutputIDInfo().getOutput(i));
                        outputRemapping[otherId] = i;
                    }

                    ONNXInitializer indices = onnx.array("ensemble_output_gather_indices", outputRemapping);

                    ONNXNode gatheredOutput = unsqueezedOutput.apply(ONNXOperators.GATHER, indices, Collections.singletonMap("axis", 1));
                    unsqueezedMembers.add(gatheredOutput);
                }
            } else {
                throw new IllegalStateException("Ensemble member '" + model.toString() + "' is not ONNXExportable.");
            }
        }

        ONNXInitializer ensembleWeights = onnx.array("ensemble_weights", weights);
        ONNXNode concat = onnx.operation(ONNXOperators.CONCAT, unsqueezedMembers, "ensemble_concat", Collections.singletonMap("axis", 2));

        return combiner.exportCombiner(concat, ensembleWeights);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        WeightedEnsembleModelProto.Builder modelBuilder = WeightedEnsembleModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        for (Model<T> m : models) {
            modelBuilder.addModels(m.serialize());
        }
        modelBuilder.addAllWeights(Util.toBoxedFloats(weights));
        modelBuilder.setCombiner(combiner.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(WeightedEnsembleModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }
}
