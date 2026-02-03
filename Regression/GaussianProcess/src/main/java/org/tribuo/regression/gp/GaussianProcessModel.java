/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.regression.gp;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.kernel.Kernel;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.regression.Regressor;
import org.tribuo.util.Util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Implements a Gaussian Process regression.
 * <p>
 * Note this implementation is not approximate and requires inverting the data matrix, so should only be
 * used for small numbers of examples.
 * <p>
 * See:
 * <pre>
 * Rasmussen C, Williams C.
 * "Gaussian Processes for Machine Learning"
 * MIT Press, 2006.
 * </pre>
 */
public final class GaussianProcessModel extends Model<Regressor> {

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final String[] dimensionNames;
    private final Kernel kernel;
    private final Matrix features;
    private final DenseMatrix alpha;
    private final DenseMatrix.CholeskyFactorization fact;
    private final DenseVector outputMeans;
    private final DenseVector outputVariances;

    GaussianProcessModel(String name, String[] dimensionNames, ModelProvenance provenance, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Regressor> outputInfo, Kernel kernel, Matrix featureMatrix, DenseMatrix alphaMatrix, DenseMatrix.CholeskyFactorization fact, DenseVector outputMeans, DenseVector outputVariances) {
        super(name, provenance, featureIDMap, outputInfo, false);
        this.dimensionNames = dimensionNames;
        this.kernel = kernel;
        this.features = featureMatrix;
        this.alpha = alphaMatrix;
        this.fact = fact;
        this.outputMeans = outputMeans;
        this.outputVariances = outputVariances;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static GaussianProcessModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        GaussianProcessModelProto proto = message.unpack(GaussianProcessModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Regressor.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a regression domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Regressor> outputDomain = (ImmutableOutputInfo<Regressor>) carrier.outputDomain();

        String[] dimensions = new String[proto.getDimensionsCount()];
        if (dimensions.length != outputDomain.size()) {
            throw new IllegalStateException("Invalid protobuf, found insufficient dimension names, expected " + outputDomain.size() + ", found " + dimensions.length);
        }
        for (int i = 0; i < dimensions.length; i++) {
            dimensions[i] = proto.getDimensions(i);
        }

        int outputSize = dimensions.length;
        int numFeatures = carrier.featureDomain().size();

        Kernel kernel = Kernel.deserialize(proto.getKernel());

        Tensor featuresTensor = Tensor.deserialize(proto.getFeatures());
        if (!(featuresTensor instanceof Matrix featureMatrix)) {
            throw new IllegalStateException("Invalid protobuf, features must be a matrix, found " + featuresTensor.getClass());
        }
        if (featureMatrix.getDimension2Size() != numFeatures) {
            throw new IllegalStateException("Invalid protobuf, feature matrix not the right size, expected [numExamples,"+ numFeatures +"], found " + Arrays.toString(featureMatrix.getShape()));
        }

        Tensor alphaTensor = Tensor.deserialize(proto.getAlphas());
        if (!(alphaTensor instanceof DenseMatrix alphaMatrix)) {
            throw new IllegalStateException("Invalid protobuf, alpha must be a dense matrix, found " + alphaTensor.getClass());
        }
        if (alphaMatrix.getDimension2Size() != outputSize) {
            throw new IllegalStateException("Invalid protobuf, alpha matrix not the right size, expected [numExamples,"+ outputSize +"], found " + Arrays.toString(alphaMatrix.getShape()));
        }

        DenseMatrix.CholeskyFactorization factorization = DenseMatrix.CholeskyFactorization.deserialize(proto.getCholesky());
        if (factorization.dim1() != outputSize) {
            throw new IllegalStateException("Invalid protobuf, cholesky not the right size, expected ["+outputSize+","+ outputSize +"], found " + Arrays.toString(factorization.lMatrix().getShape()));
        }

        Tensor outputMeansTensor = Tensor.deserialize(proto.getOutputMeans());
        if (!(outputMeansTensor instanceof DenseVector outputMeans)) {
            throw new IllegalStateException("Invalid protobuf, output means must be a dense vector, found " + outputMeansTensor.getClass());
        }
        if (outputMeans.size() != outputSize) {
            throw new IllegalStateException("Invalid protobuf, output means not the right size, expected " + outputSize + ", found " + outputMeans.size());
        }
        Tensor outputVariancesTensor = Tensor.deserialize(proto.getOutputVariances());
        if (!(outputVariancesTensor instanceof DenseVector outputVariances)) {
            throw new IllegalStateException("Invalid protobuf, output variances must be a dense vector, found " + outputVariancesTensor.getClass());
        }
        if (outputVariances.size() != outputSize) {
            throw new IllegalStateException("Invalid protobuf, output variances not the right size, expected " + outputSize + ", found " + outputVariances.size());
        }

        return new GaussianProcessModel(carrier.name(), dimensions, carrier.provenance(), carrier.featureDomain(), outputDomain,
                kernel, featureMatrix, alphaMatrix, factorization, outputMeans, outputVariances);
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<Regressor> carrier = createDataCarrier();

        GaussianProcessModelProto.Builder modelBuilder = GaussianProcessModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.addAllDimensions(Arrays.asList(dimensionNames));
        modelBuilder.setKernel(kernel.serialize());
        modelBuilder.setFeatures(features.serialize());
        modelBuilder.setAlphas(alpha.serialize());
        modelBuilder.setFactorization(fact.lMatrix().serialize());
        modelBuilder.setOutputMeans(outputMeans.serialize());
        modelBuilder.setOutputVariances(outputVariances.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(GaussianProcessModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    public Prediction<Regressor> predict(Example<Regressor> example) {
        SGDVector vec;
        if (example.size() == featureIDMap.size()) {
            vec = DenseVector.createDenseVector(example, featureIDMap, false);
        } else {
            vec = SparseVector.createSparseVector(example, featureIDMap, false);
        }
        DenseVector sim = kernel.computeSimilarityVector(vec, features);
        DenseVector meanPred = alpha.rightMultiply(sim);
        meanPred.hadamardProductInPlace(outputVariances);
        meanPred.intersectAndAddInPlace(outputMeans);
        return new Prediction<>(new Regressor(dimensionNames,meanPred.toArray()), vec.numActiveElements(), example);
    }

    @Override
    protected List<Prediction<Regressor>> innerPredict(Iterable<Example<Regressor>> examples) {
        List<Example<Regressor>> exampleList = new ArrayList<>();
        List<SGDVector> vectors = new ArrayList<>();
        for (Example<Regressor> example : examples) {
            if (example.size() == featureIDMap.size()) {
                vectors.add(DenseVector.createDenseVector(example, featureIDMap, false));
            } else {
                vectors.add(SparseVector.createSparseVector(example, featureIDMap, false));
            }
            exampleList.add(example);
        }
        Matrix mat = Matrix.aggregate(vectors.toArray(new SGDVector[0]), vectors.size(), false);
        DenseMatrix sim = kernel.computeSimilarityMatrix(mat, features);
        DenseMatrix meanPred = sim.matrixMultiply(alpha);
        meanPred.rowHadamardProductInPlace(outputVariances);
        meanPred.rowIntersectAndAddInPlace(outputMeans);
        List<Prediction<Regressor>> predictions = new ArrayList<>();
        for (int i = 0; i < mat.getDimension1Size(); i++) {
            var pred = new Prediction<>(new Regressor(dimensionNames,meanPred.getRow(i).toArray()), mat.getRow(i).numActiveElements(), exampleList.get(i));
            predictions.add(pred);
        }
        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<Regressor>> getExcuse(Example<Regressor> example) {
        return Optional.empty();
    }

    @Override
    protected Model<Regressor> copy(String newName, ModelProvenance newProvenance) {
        return new GaussianProcessModel(newName, Arrays.copyOf(dimensionNames, dimensionNames.length), newProvenance,
                featureIDMap, outputIDInfo, kernel, features.copy(), alpha.copy(), fact.copy(), outputMeans.copy(),
                outputVariances.copy());
    }
}
