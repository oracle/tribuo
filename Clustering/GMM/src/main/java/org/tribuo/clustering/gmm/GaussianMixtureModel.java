/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.clustering.gmm;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.clustering.ClusterID;
import org.tribuo.impl.ArrayExample;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.random.RandomGenerator;

/**
 * A Gaussian Mixture Model.
 * <p>
 * The predict method of this model assigns the provided input to a cluster,
 * but it does not update the model's centroids.
 * <p>
 * The predict method is single threaded.
 * <p>
 * See:
 * <pre>
 * J. Friedman, T. Hastie, &amp; R. Tibshirani.
 * "The Elements of Statistical Learning"
 * Springer 2001. <a href="http://web.stanford.edu/~hastie/ElemStatLearn/">PDF</a>
 * </pre>
 */
public class GaussianMixtureModel extends Model<ClusterID> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final DenseVector[] meanVectors;

    private final DenseMatrix[] covarianceMatrices;

    private final DenseVector mixingDistribution;

    GaussianMixtureModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap,
                         ImmutableOutputInfo<ClusterID> outputIDInfo, DenseVector[] meanVectors,
                         DenseMatrix[] covarianceMatrices, DenseVector mixingDistribution) {
        super(name,description,featureIDMap,outputIDInfo,false);
        this.meanVectors = meanVectors;
        this.covarianceMatrices = covarianceMatrices;
        this.mixingDistribution = mixingDistribution;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static GaussianMixtureModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        GaussianMixtureModelProto proto = message.unpack(GaussianMixtureModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(ClusterID.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a clustering domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<ClusterID> outputDomain = (ImmutableOutputInfo<ClusterID>) carrier.outputDomain();

        ImmutableFeatureMap featureDomain = carrier.featureDomain();

        final int means = proto.getMeanVectorsCount();

        if (means == 0) {
            throw new IllegalStateException("Invalid protobuf, no centroids were found");
        } else if (proto.getCovarianceMatricesCount() != means) {
            throw new IllegalStateException("Invalid protobuf, found " + means + " means, but " + proto.getCovarianceMatricesCount() + " covariances.");
        }
        DenseVector[] centroids = new DenseVector[means];
        List<TensorProto> centroidProtos = proto.getMeanVectorsList();
        for (int i = 0; i < centroids.length; i++) {
            Tensor centroidTensor = Tensor.deserialize(centroidProtos.get(i));
            if (centroidTensor instanceof DenseVector centroid) {
                if (centroid.size() != featureDomain.size()) {
                    throw new IllegalStateException("Invalid protobuf, centroid did not contain all the features, found " + centroid.size() + " expected " + featureDomain.size());
                }
                centroids[i] = centroid;
            } else {
                throw new IllegalStateException("Invalid protobuf, expected centroid to be a dense vector, found " + centroidTensor.getClass());
            }
        }
        DenseMatrix[] covariances = new DenseMatrix[means];
        List<TensorProto> covarianceProtos = proto.getCovarianceMatricesList();
        for (int i = 0; i < covariances.length; i++) {
            Tensor covarianceTensor = Tensor.deserialize(covarianceProtos.get(i));
            if (covarianceTensor instanceof DenseMatrix covariance) {
                if (covariance.getDimension1Size() != featureDomain.size() || covariance.getDimension2Size() != featureDomain.size()) {
                    throw new IllegalStateException("Invalid protobuf, covariance was not square or did not contain all " +
                            "the features, found " + Arrays.toString(covariance.getShape()) + " expected [" + featureDomain.size() + ", " + featureDomain.size() +"]");
                }
                covariances[i] = covariance;
            } else {
                throw new IllegalStateException("Invalid protobuf, expected covariance to be a dense matrix, found " + covarianceTensor.getClass());
            }
        }
        Tensor mixing = Tensor.deserialize(proto.getMixingDistribution());
        DenseVector mixingVec;
        if (mixing instanceof DenseVector mixingDist) {
            if (mixingDist.size() != means) {
                throw new IllegalStateException("Invalid protobuf, found " + means + " but a " + mixingDist.size() + " element mixing distribution");
            } else {
                mixingVec = mixingDist;
            }
        } else {
            throw new IllegalStateException("Invalid protobuf, expected mixing distribution to be a dense vector, found " + mixing.getClass());
        }

        return new GaussianMixtureModel(carrier.name(), carrier.provenance(), featureDomain, outputDomain, centroids, covariances, mixingVec);
    }

    /**
     * Returns a copy of the centroids.
     * <p>
     * In most cases you should prefer {@link #getMeans} as
     * it performs the mapping from Tribuo's internal feature ids
     * to the externally visible feature names for you.
     * This method provides direct access to the centroid vectors
     * for use in downstream processing if the ids are not relevant
     * (or are known to match).
     * @return The centroids.
     */
    public DenseVector[] getMeanVectors() {
        DenseVector[] copies = new DenseVector[meanVectors.length];

        for (int i = 0; i < copies.length; i++) {
            copies[i] = meanVectors[i].copy();
        }

        return copies;
    }

    /**
     * Returns a copy of the covariances.
     * <p>
     * This method provides direct access to the covariance matrices
     * for use in downstream processing, users need to map the indices using
     * Tribuo's internal ids themselves.
     * @return The covariances.
     */
    public DenseMatrix[] getCovariances() {
        DenseMatrix[] copies = new DenseMatrix[covarianceMatrices.length];

        for (int i = 0; i < copies.length; i++) {
            copies[i] = covarianceMatrices[i].copy();
        }

        return copies;
    }

    /**
     * Returns a list of features, one per centroid.
     * <p>
     * This should be used in preference to {@link #getMeanVectors()}
     * as it performs the mapping from Tribuo's internal feature ids to
     * the externally visible feature names.
     * </p>
     * @return A list containing all the centroids.
     */
    public List<List<Feature>> getMeans() {
        List<List<Feature>> output = new ArrayList<>(meanVectors.length);

        for (int i = 0; i < meanVectors.length; i++) {
            List<Feature> features = new ArrayList<>(featureIDMap.size());

            for (VectorTuple v : meanVectors[i]) {
                Feature f = new Feature(featureIDMap.get(v.index).getName(),v.value);
                features.add(f);
            }

            output.add(features);
        }

        return output;
    }

    @Override
    public Prediction<ClusterID> predict(Example<ClusterID> example) {
        SGDVector vector;
        if (example.size() == featureIDMap.size()) {
            vector = DenseVector.createDenseVector(example, featureIDMap, false);
        } else {
            vector = SparseVector.createSparseVector(example, featureIDMap, false);
        }
        if (vector.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        // generate cluster responsibilities and normalize into a distribution

        return new Prediction<>(new ClusterID(id),vector.size(),example);
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    public Optional<Excuse<ClusterID>> getExcuse(Example<ClusterID> example) {
        return Optional.empty();
    }

    /**
     * Sample from this Gaussian Mixture Model.
     * @param numSamples The number of samples to draw.
     * @param rng The RNG to use.
     * @return A list of samples from this GMM.
     */
    public List<Pair<Integer, DenseVector>> sample(int numSamples, RandomGenerator rng) {
        // Convert mixing distribution into CDF

        // Sample from mixing distribution

        // Sample from appropriate MultivariateNormalDistribution

    }

    /**
     * Sample from this Gaussian Mixture Model.
     * @param numSamples The number of samples to draw.
     * @param rng The RNG to use.
     * @return A list of examples sampled from this GMM.
     */
    public List<Example<ClusterID>> sampleExamples(int numSamples, RandomGenerator rng) {
        var samples = sample(numSamples, rng);

        List<Example<ClusterID>> output = new ArrayList<>();

        for (Pair<Integer, DenseVector> e : samples) {
            ClusterID id = outputIDInfo.getOutput(e.getA());
            String[] names = new String[e.getB().size()];
            double[] values = new double[e.getB().size()];
            for (VectorTuple v : e.getB()) {
                names[v.index] = featureIDMap.get(v.index).getName();
                values[v.index] = v.value;
            }
            Example<ClusterID> ex = new ArrayExample<>(id, names, values);
            output.add(ex);
        }

        return output;
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<ClusterID> carrier = createDataCarrier();

        GaussianMixtureModelProto.Builder modelBuilder = GaussianMixtureModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        for (DenseVector e : meanVectors) {
            modelBuilder.addMeanVectors(e.serialize());
        }
        for (DenseMatrix e : covarianceMatrices) {
            modelBuilder.addCovarianceMatrices(e.serialize());
        }
        modelBuilder.setMixingDistribution(mixingDistribution.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(GaussianMixtureModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected GaussianMixtureModel copy(String newName, ModelProvenance newProvenance) {
        DenseVector[] newCentroids = new DenseVector[meanVectors.length];
        DenseMatrix[] newCovariance = new DenseMatrix[meanVectors.length];
        for (int i = 0; i < meanVectors.length; i++) {
            newCentroids[i] = meanVectors[i].copy();
            newCovariance[i] = covarianceMatrices[i].copy();
        }

        return new GaussianMixtureModel(newName,newProvenance,featureIDMap,outputIDInfo,
                newCentroids,newCovariance,mixingDistribution.copy());
    }
}
