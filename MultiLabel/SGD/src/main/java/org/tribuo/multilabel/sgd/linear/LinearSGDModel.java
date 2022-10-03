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

package org.tribuo.multilabel.sgd.linear;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.util.VectorNormalizer;
import org.tribuo.multilabel.MultiLabel;
import org.tribuo.multilabel.sgd.protos.MultiLabelLinearSGDProto;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.onnx.ONNXNode;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The inference time version of a multi-label linear model trained using SGD.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<MultiLabel> implements ONNXExportable {
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final VectorNormalizer normalizer;
    private final double threshold;

    /**
     * Constructs a linear regression model trained via SGD.
     * @param name The model name.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters (i.e., the weight matrix).
     * @param normalizer The output normalizer (usually sigmoid or no-op).
     * @param generatesProbabilities Does this model produce probabilistic outputs.
     * @param threshold The threshold for emitting a label.
     */
    LinearSGDModel(String name, ModelProvenance provenance,
                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<MultiLabel> outputIDInfo,
                   LinearParameters parameters, VectorNormalizer normalizer, boolean generatesProbabilities, double threshold) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, generatesProbabilities);
        this.normalizer = normalizer;
        this.threshold = threshold;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LinearSGDModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        MultiLabelLinearSGDProto proto = message.unpack(MultiLabelLinearSGDProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(MultiLabel.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a multi-label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<MultiLabel> outputDomain = (ImmutableOutputInfo<MultiLabel>) carrier.outputDomain();

        Parameters params = Parameters.deserialize(proto.getParams());
        if (!(params instanceof LinearParameters)) {
            throw new IllegalStateException("Invalid protobuf, parameters must be LinearParameters, found " + params.getClass());
        }

        VectorNormalizer normalizer = VectorNormalizer.deserialize(proto.getNormalizer());

        return new LinearSGDModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,
                (LinearParameters) params, normalizer, carrier.generatesProbabilities(), proto.getThreshold());
    }

    @Override
    public Prediction<MultiLabel> predict(Example<MultiLabel> example) {
        PredAndActive predTuple = predictSingle(example);
        DenseVector outputs = predTuple.prediction;
        outputs.normalize(normalizer);
        Map<String,MultiLabel> fullLabels = new HashMap<>();
        Set<Label> predictedLabels = new HashSet<>();
        for (int i = 0; i < outputs.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabelString();
            double labelScore = outputs.get(i);
            Label score = new Label(outputIDInfo.getOutput(i).getLabelString(),labelScore);
            if (labelScore > threshold) {
                predictedLabels.add(score);
            }
            fullLabels.put(labelName,new MultiLabel(score));
        }
        return new Prediction<>(new MultiLabel(predictedLabels), fullLabels, predTuple.numActiveFeatures - 1, example, generatesProbabilities);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<MultiLabel> carrier = createDataCarrier();
        MultiLabelLinearSGDProto.Builder modelBuilder = MultiLabelLinearSGDProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setParams(modelParameters.serialize());
        modelBuilder.setNormalizer(normalizer.serialize());
        modelBuilder.setThreshold(threshold);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(LinearSGDModel.class.getName());
        builder.setSerializedData(Any.pack(modelBuilder.build()));

        return builder.build();
    }

    @Override
    protected String getDimensionName(int index) {
        return outputIDInfo.getOutput(index).getLabelString();
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,newProvenance,featureIDMap,outputIDInfo,(LinearParameters)modelParameters.copy(),normalizer,generatesProbabilities,threshold);
    }

    @Override
    protected ONNXNode onnxOutput(ONNXNode input) {
        return normalizer.exportNormalizer(input);
    }

    @Override
    protected String onnxModelName() {
        return "MultiLabel-LinearSGDModel";
    }
}
