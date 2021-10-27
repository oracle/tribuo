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

package org.tribuo.ensemble;

import ai.onnx.proto.OnnxMl;
import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.Tribuo;
import org.tribuo.onnx.ONNXContext;
import org.tribuo.onnx.ONNXExportable;
import org.tribuo.onnx.ONNXOperators;
import org.tribuo.onnx.ONNXShape;
import org.tribuo.onnx.ONNXUtils;
import org.tribuo.provenance.EnsembleModelProvenance;
import org.tribuo.provenance.impl.TimestampedTrainerProvenance;
import org.tribuo.util.Util;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;

/**
 * An ensemble model that uses weights to combine the ensemble member predictions.
 */
public final class WeightedEnsembleModel<T extends Output<T>> extends EnsembleModel<T> implements ONNXExportable {
    private static final long serialVersionUID = 1L;

    protected final float[] weights;

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
                          ImmutableOutputInfo<T> outputIDInfo,
                          List<Model<T>> newModels, EnsembleCombiner<T> combiner, float[] weights) {
        super(name,provenance,featureIDMap,outputIDInfo,newModels);
        this.weights = Arrays.copyOf(weights,weights.length);
        this.combiner = combiner;
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

        // Validate output domains
        ImmutableOutputInfo<T> outputInfo = models.get(0).getOutputIDInfo();
        List<Pair<Integer,T>> firstList = new ArrayList<>();
        for (Pair<Integer,T> p : outputInfo) {
            firstList.add(p);
        }
        List<Pair<Integer,T>> comparisonList = new ArrayList<>();
        for (int i = 1; i < models.size(); i++) {
            comparisonList.clear();
            for (Pair<Integer,T> p : models.get(i).getOutputIDInfo()) {
                comparisonList.add(p);
            }
            if (!firstList.equals(comparisonList)) {
                throw new IllegalArgumentException("Model output domains are not equal.");
            }
        }

        // Extract feature domain
        ImmutableFeatureMap featureMap = models.get(0).getFeatureIDMap();

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
        ONNXContext context = new ONNXContext();

        // Make inputs and outputs
        OnnxMl.TypeProto inputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,featureIDMap.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto inputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(inputType).setName("input").build();
        context.addInput(inputValueProto);
        OnnxMl.TypeProto outputType = ONNXUtils.buildTensorTypeNode(new ONNXShape(new long[]{-1,outputIDInfo.size()}, new String[]{"batch",null}), OnnxMl.TensorProto.DataType.FLOAT);
        OnnxMl.ValueInfoProto outputValueProto = OnnxMl.ValueInfoProto.newBuilder().setType(outputType).setName("output").build();
        context.addOutput(outputValueProto);
        context.setName("WeightedEnsembleModel");

        // Build graph
        writeONNXGraph(context, inputValueProto.getName(), outputValueProto.getName());

        // Build model
        OnnxMl.ModelProto.Builder builder = OnnxMl.ModelProto.newBuilder();
        builder.setGraph(context.buildGraph());
        builder.setDomain(domain);
        builder.setProducerName("Tribuo");
        builder.setProducerVersion(Tribuo.VERSION);
        builder.setModelVersion(modelVersion);
        builder.setDocString(toString());
        builder.addOpsetImport(ONNXOperators.getOpsetProto());
        builder.setIrVersion(6);

        // Extract provenance and store in metadata
        OnnxMl.StringStringEntryProto.Builder metaBuilder = OnnxMl.StringStringEntryProto.newBuilder();
        metaBuilder.setKey(ONNXExportable.PROVENANCE_METADATA_FIELD);
        metaBuilder.setValue(serializeProvenance(getProvenance()));
        builder.addMetadataProps(metaBuilder.build());

        return builder.build();
    }

    @Override
    public void writeONNXGraph(ONNXContext context, String inputName, String outputName) {
        String[] memberOutputs = new String[models.size()];
        // Write in each ensemble member
        for (int i = 0; i < models.size(); i++) {
            Model<T> model = models.get(i);
            if (model instanceof ONNXExportable) {
                memberOutputs[i] = context.generateUniqueName("ensemble_member_output");
                ((ONNXExportable)model).writeONNXGraph(context,inputName,memberOutputs[i]);
            } else {
                throw new IllegalStateException("Ensemble member '" + model.toString() + "' is not ONNXExportable.");
            }
        }

        // Unsqueeze the members so they have an extra dimension to concatenate along
        OnnxMl.TensorProto unsqueezeAxes = ONNXUtils.arrayBuilder(context,"unsqueeze_ensemble_output",new long[]{2});
        context.addInitializer(unsqueezeAxes);
        String[] unsqueezeOutputs = new String[memberOutputs.length];
        for (int i = 0; i < unsqueezeOutputs.length; i++) {
            unsqueezeOutputs[i] = context.generateUniqueName("unsqueeze_output");
            OnnxMl.NodeProto unsqueeze = ONNXOperators.UNSQUEEZE.build(context,new String[]{memberOutputs[i],unsqueezeAxes.getName()},unsqueezeOutputs[i]);
            context.addNode(unsqueeze);
        }

        // Concatenate all the outputs into a single three tensor [batch_size, num_outputs, num_members]
        OnnxMl.NodeProto concat = ONNXOperators.CONCAT.build(context,unsqueezeOutputs,context.generateUniqueName("ensemble_concat"), Collections.singletonMap("axis",2));
        context.addNode(concat);

        // Write in weights initializer
        OnnxMl.TensorProto weightTensor = ONNXUtils.arrayBuilder(context,"ensemble_weights",weights);
        context.addInitializer(weightTensor);

        // Write in ensemble combiner
        List<OnnxMl.NodeProto> combinerNodes = combiner.exportCombiner(context,concat.getOutput(0),outputName,weightTensor.getName());
        if (combinerNodes.isEmpty()) {
            throw new IllegalStateException("This ensemble cannot be exported as the combiner '" + combiner.getClass() + "' uses the default implementation of EnsembleCombiner.exportCombiner.");
        }
        context.addAllNodes(combinerNodes);
    }
}
