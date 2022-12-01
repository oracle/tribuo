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

package org.tribuo.multilabel.baseline;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.protos.ClassifierChainModelProto;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.OutputProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_NEGATIVE;
import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_POSITIVE;
import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_PREFIX;
import static org.tribuo.multilabel.baseline.ClassifierChainTrainer.CC_SEPARATOR;

/**
 * A Classifier Chain Model.
 * <p>
 * Classifier chains convert binary classifiers into multi-label
 * classifiers by training one classifier per label (similar to
 * the Binary Relevance approach), but in a specific order (the chain).
 * Classifiers further down the chain use the labels from all previously
 * computed classifiers as features, thus allowing the model to incorporate
 * some measure of label dependence.
 * <p>
 * Choosing the optimal label ordering is tricky as the label dependence
 * is usually unknown, so one popular alternative is to produce an ensemble
 * of randomly ordered chains, which mitigates a poor label ordering by averaging
 * across many orderings.
 * <p>
 * See:
 * <pre>
 * Read, J., Pfahringer, B., Holmes, G., &amp; Frank, E.
 * "Classifier Chains for Multi-Label Classification"
 * Machine Learning, pages 333-359, 2011.
 * </pre>
 */
public final class ClassifierChainModel extends Model<MultiLabel> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final List<Model<Label>> models;
    private final List<Label> labelOrder;

    /**
     * The list of Label and list of Model must be in the same order, and have a bijection.
     * @param labelOrder The list of labels this model was trained on.
     * @param models The list of individual binary models.
     * @param description A description of the trainer.
     * @param featureMap The feature domain used in training.
     * @param labelInfo The label domain used in training.
     */
    ClassifierChainModel(List<Label> labelOrder, List<Model<Label>> models, ModelProvenance description, ImmutableFeatureMap featureMap, ImmutableOutputInfo<MultiLabel> labelInfo) {
        super("classifier-chain",description,featureMap,labelInfo,false);
        this.labelOrder = Collections.unmodifiableList(labelOrder);
        this.models = Collections.unmodifiableList(models);
    }

    private ClassifierChainModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap,
                                 ImmutableOutputInfo<MultiLabel> labelInfo, List<Label> labelOrder,
                                 List<Model<Label>> models) {
        super(name,provenance,featureMap,labelInfo,false);
        this.labelOrder = Collections.unmodifiableList(labelOrder);
        this.models = Collections.unmodifiableList(models);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static ClassifierChainModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        ClassifierChainModelProto proto = message.unpack(ClassifierChainModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(MultiLabel.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a multi-label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<MultiLabel> outputDomain = (ImmutableOutputInfo<MultiLabel>) carrier.outputDomain();

        if (proto.getLabelOrderCount() != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, mismatch in number of labels, found " + proto.getLabelOrderCount() + " expected " + outputDomain.size());
        }
        if (proto.getLabelOrderCount() != proto.getModelsCount()) {
            throw new IllegalStateException("Invalid protobuf, expected one model per label, found " + proto.getModelsCount() + " models and " + outputDomain.size() + " labels");
        }
        List<Label> labelOrder = new ArrayList<>(proto.getLabelOrderCount());
        for (OutputProto p : proto.getLabelOrderList()) {
            Output<?> output = Output.deserialize(p);
            if (output instanceof Label) {
                labelOrder.add((Label)output);
            } else {
                throw new IllegalStateException("Invalid protobuf, expected label ordering, found " + output.getClass());
            }
        }
        List<Model<Label>> models = new ArrayList<>(proto.getModelsCount());
        for (ModelProto p : proto.getModelsList()) {
            Model<?> model = Model.deserialize(p);
            if (model.validate(Label.class)) {
                models.add(model.castModel(Label.class));
            } else {
                throw new IllegalStateException("Invalid protobuf, expected all models to be classification, found " + model);
            }
        }

        return new ClassifierChainModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,labelOrder,models);
    }

    /**
     * Returns an unmodifiable view on the chain members.
     * @return The chain members.
     */
    public List<Model<Label>> getModels() {
        return models;
    }

    @Override
    public Prediction<MultiLabel> predict(Example<MultiLabel> example) {
        Set<Label> predictedLabels = new HashSet<>();
        BinaryExample e = new BinaryExample(example,MultiLabel.NEGATIVE_LABEL);
        int numUsed = 0;
        for (int i = 0; i < labelOrder.size(); i++) {
            Model<Label> curModel = models.get(i);
            Label curLabel = labelOrder.get(i);
            Prediction<Label> p = curModel.predict(e);
            if (numUsed < p.getNumActiveFeatures()) {
                numUsed = p.getNumActiveFeatures();
            }
            String featureName;
            if (!p.getOutput().getLabel().equals(MultiLabel.NEGATIVE_LABEL_STRING)) {
                predictedLabels.add(p.getOutput());
                // update example with positive label feature.
                featureName = CC_PREFIX + CC_SEPARATOR + curLabel.getLabel() + CC_SEPARATOR + CC_POSITIVE;
            } else {
                // update example with negative label feature.
                featureName = CC_PREFIX + CC_SEPARATOR + curLabel.getLabel() + CC_SEPARATOR + CC_NEGATIVE;
            }
            e.add(new Feature(featureName,1.0));
        }
        return new Prediction<>(new MultiLabel(predictedLabels),numUsed,example);
    }

    /**
     * Returns the training label order.
     * @return The training label order.
     */
    public List<Label> getLabelOrder() {
        return labelOrder;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<MultiLabel>> getExcuse(Example<MultiLabel> example) {
        return Optional.empty();
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<MultiLabel> carrier = createDataCarrier();

        ClassifierChainModelProto.Builder modelBuilder = ClassifierChainModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        for (Model<Label> m : models) {
            modelBuilder.addModels(m.serialize());
        }
        for (Label l : labelOrder) {
            modelBuilder.addLabelOrder(l.serialize());
        }

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(ClassifierChainModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected ClassifierChainModel copy(String newName, ModelProvenance newProvenance) {
        List<Model<Label>> newModels = new ArrayList<>();
        for (Model<Label> e : models) {
            newModels.add(e.copy());
        }
        return new ClassifierChainModel(labelOrder,newModels,newProvenance,featureIDMap,outputIDInfo);
    }
}
