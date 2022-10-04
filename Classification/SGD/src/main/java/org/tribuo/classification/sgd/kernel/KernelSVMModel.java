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

package org.tribuo.classification.sgd.kernel;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.protos.KernelSVMModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The inference time version of a kernel model trained using Pegasos.
 * <p>
 * See:
 * <pre>
 * Shalev-Shwartz S, Singer Y, Srebro N, Cotter A
 * "Pegasos: Primal Estimated Sub-Gradient Solver for SVM"
 * Mathematical Programming, 2011.
 * </pre>
 */
public class KernelSVMModel extends Model<Label> {
    private static final long serialVersionUID = 2L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final Kernel kernel;
    private final SparseVector[] supportVectors;
    private final DenseMatrix weights;

    KernelSVMModel(String name, ModelProvenance description,
                          ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Label> labelIDMap,
                          Kernel kernel, SparseVector[] supportVectors, DenseMatrix weights) {
        super(name, description, featureIDMap, labelIDMap, false);
        this.kernel = kernel;
        this.supportVectors = supportVectors;
        this.weights = weights;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static KernelSVMModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        KernelSVMModelProto proto = message.unpack(KernelSVMModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Label.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a label domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Label> outputDomain = (ImmutableOutputInfo<Label>) carrier.outputDomain();

        SparseVector[] supportVectors = new SparseVector[proto.getSupportVectorsCount()];
        int featureSize = carrier.featureDomain().size() + 1;
        List<TensorProto> supportProtos = proto.getSupportVectorsList();
        for (int i = 0; i < supportProtos.size(); i++) {
            Tensor tensor = Tensor.deserialize(supportProtos.get(i));
            if (!(tensor instanceof SparseVector)) {
                throw new IllegalStateException("Invalid protobuf, support vector must be a sparse vector, found " + tensor.getClass());
            }
            SparseVector vec = (SparseVector) tensor;
            if (vec.size() != featureSize) {
                throw new IllegalStateException("Invalid protobuf, support vector size must equal feature domain size, found " + vec.size() + ", expected " + featureSize);
            }
            supportVectors[i] = vec;
        }

        Tensor weightTensor = Tensor.deserialize(proto.getWeights());
        if (!(weightTensor instanceof DenseMatrix)) {
            throw new IllegalStateException("Invalid protobuf, weights must be a dense matrix, found " + weightTensor.getClass());
        }
        DenseMatrix weights = (DenseMatrix) weightTensor;
        if (weights.getDimension1Size() != carrier.outputDomain().size()) {
            throw new IllegalStateException("Invalid protobuf, weights not the right size, expected " + carrier.outputDomain().size() + ", found " + weights.getDimension1Size());
        }
        if (weights.getDimension2Size() != supportVectors.length) {
            throw new IllegalStateException("Invalid protobuf, weights not the right size, expected " + supportVectors.length + ", found " + weights.getDimension2Size());
        }

        Kernel kernel = Kernel.deserialize(proto.getKernel());

        return new KernelSVMModel(carrier.name(), carrier.provenance(), carrier.featureDomain(), outputDomain,
                kernel, supportVectors, weights);
    }

    /**
     * Returns the number of support vectors used.
     * @return The number of support vectors.
     */
    public int getNumberOfSupportVectors() {
        return supportVectors.length;
    }

    @Override
    public Prediction<Label> predict(Example<Label> example) {
        SparseVector features = SparseVector.createSparseVector(example,featureIDMap,true);
        // Due to bias feature
        if (features.numActiveElements() == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }
        double[] scores = new double[supportVectors.length];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = kernel.similarity(features,supportVectors[i]);
        }
        DenseVector scoreVector = DenseVector.createDenseVector(scores);
        DenseVector prediction = weights.leftMultiply(scoreVector);

        double maxScore = Double.NEGATIVE_INFINITY;
        Label maxLabel = null;
        Map<String,Label> predMap = new LinkedHashMap<>();
        for (int i = 0; i < prediction.size(); i++) {
            String labelName = outputIDInfo.getOutput(i).getLabel();
            Label label = new Label(labelName, prediction.get(i));
            predMap.put(labelName, label);
            if (label.getScore() > maxScore) {
                maxScore = label.getScore();
                maxLabel = label;
            }
        }
        return new Prediction<>(maxLabel, predMap, features.numActiveElements(), example, generatesProbabilities);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<Label>> getExcuse(Example<Label> example) {
        return Optional.empty();
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Label> carrier = createDataCarrier();
        KernelSVMModelProto.Builder modelBuilder = KernelSVMModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setKernel(kernel.serialize());
        modelBuilder.setWeights(weights.serialize());
        for (SparseVector v : supportVectors) {
            modelBuilder.addSupportVectors(v.serialize());
        }

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setVersion(CURRENT_VERSION);
        builder.setClassName(KernelSVMModel.class.getName());
        builder.setSerializedData(Any.pack(modelBuilder.build()));

        return builder.build();
    }

    @Override
    protected KernelSVMModel copy(String newName, ModelProvenance newProvenance) {
        SparseVector[] vectorCopies = new SparseVector[supportVectors.length];
        for (int i = 0; i < vectorCopies.length; i++) {
            vectorCopies[i] = supportVectors[i].copy();
        }
        return new KernelSVMModel(newName,newProvenance,featureIDMap,outputIDInfo,kernel,vectorCopies,new DenseMatrix(weights));
    }
}
