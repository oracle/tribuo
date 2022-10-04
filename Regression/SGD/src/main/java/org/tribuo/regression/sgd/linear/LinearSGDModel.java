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

package org.tribuo.regression.sgd.linear;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.ONNXExportable;
import org.tribuo.Prediction;
import org.tribuo.common.sgd.AbstractLinearSGDModel;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.LinearParameters;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.regression.sgd.protos.RegressionLinearSGDProto;
import org.tribuo.util.onnx.ONNXNode;

import java.io.IOException;
import java.util.Arrays;

/**
 * The inference time version of a linear model trained using SGD.
 * The output dimensions are independent, unless they are tied together by the
 * optimiser.
 * <p>
 * See:
 * <pre>
 * Bottou L.
 * "Large-Scale Machine Learning with Stochastic Gradient Descent"
 * Proceedings of COMPSTAT, 2010.
 * </pre>
 */
public class LinearSGDModel extends AbstractLinearSGDModel<Regressor> implements ONNXExportable {
    private static final long serialVersionUID = 3L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final String[] dimensionNames;

    // Unused as the weights now live in AbstractSGDModel
    // It remains for serialization compatibility with Tribuo 4.0
    @Deprecated
    private DenseMatrix weights = null;

    /**
     * Constructs a linear regression model trained via SGD.
     * @param name The model name.
     * @param dimensionNames The regression dimension names.
     * @param provenance The model provenance.
     * @param featureIDMap The feature domain.
     * @param outputIDInfo The output domain.
     * @param parameters The model parameters (i.e., the weight matrix).
     */
    LinearSGDModel(String name, String[] dimensionNames, ModelProvenance provenance,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputIDInfo,
                          LinearParameters parameters) {
        super(name, provenance, featureIDMap, outputIDInfo, parameters, false);
        this.dimensionNames = dimensionNames;
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
        RegressionLinearSGDProto proto = message.unpack(RegressionLinearSGDProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Regressor.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a regression domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Regressor> outputDomain = (ImmutableOutputInfo<Regressor>) carrier.outputDomain();

        Parameters params = Parameters.deserialize(proto.getParams());
        if (!(params instanceof LinearParameters)) {
            throw new IllegalStateException("Invalid protobuf, parameters must be LinearParameters, found " + params.getClass());
        }

        String[] dimensionNames = proto.getDimensionNamesList().toArray(new String[0]);
        if (dimensionNames.length != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, found a different number of dimension names to the output dimensions, found " + dimensionNames.length + " , expected " + outputDomain.size());
        }

        return new LinearSGDModel(carrier.name(), dimensionNames, carrier.provenance(), carrier.featureDomain(),
                outputDomain, (LinearParameters) params);
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        PredAndActive predTuple = predictSingle(example);
        return new Prediction<>(new Regressor(dimensionNames,predTuple.prediction.toArray()), predTuple.numActiveFeatures-1, example);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Regressor> carrier = createDataCarrier();
        RegressionLinearSGDProto.Builder modelBuilder = RegressionLinearSGDProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setParams(modelParameters.serialize());
        modelBuilder.addAllDimensionNames(Arrays.asList(dimensionNames));

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(LinearSGDModel.class.getName());
        builder.setSerializedData(Any.pack(modelBuilder.build()));

        return builder.build();
    }

    @Override
    protected LinearSGDModel copy(String newName, ModelProvenance newProvenance) {
        return new LinearSGDModel(newName,Arrays.copyOf(dimensionNames,dimensionNames.length),newProvenance,featureIDMap,outputIDInfo,(LinearParameters)modelParameters.copy());
    }

    @Override
    protected String getDimensionName(int index) {
        return dimensionNames[index];
    }

    @Override
    protected ONNXNode onnxOutput(ONNXNode input) {
        return input;
    }

    @Override
    protected String onnxModelName() {
        return "Regression-LinearSGDModel";
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();

        // Bounce old 4.0 style models into the new 4.1 style models
        if (weights != null && modelParameters == null) {
            modelParameters = new LinearParameters(weights);
            weights = null;
            addBias = true;
        }
    }
}
