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

package org.tribuo.clustering.kmeans;

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
import org.tribuo.clustering.kmeans.KMeansTrainer.Distance;
import org.tribuo.clustering.kmeans.protos.KMeansModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A K-Means model with a selectable distance function.
 * <p>
 * The predict method of this model assigns centres to the provided input,
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
public class KMeansModel extends Model<ClusterID> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final DenseVector[] centroidVectors;

    @Deprecated
    private Distance distanceType;

    // This is not final to support deserialization of older models. It will be final in a future version which doesn't
    // maintain serialization compatibility with 4.X.
    private org.tribuo.math.distance.Distance dist;

    KMeansModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap,
                ImmutableOutputInfo<ClusterID> outputIDInfo, DenseVector[] centroidVectors, org.tribuo.math.distance.Distance dist) {
        super(name,description,featureIDMap,outputIDInfo,false);
        this.centroidVectors = centroidVectors;
        this.dist = dist;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static KMeansModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        KMeansModelProto proto = message.unpack(KMeansModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(ClusterID.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a clustering domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<ClusterID> outputDomain = (ImmutableOutputInfo<ClusterID>) carrier.outputDomain();

        ImmutableFeatureMap featureDomain = carrier.featureDomain();

        if (proto.getCentroidVectorsCount() == 0) {
            throw new IllegalStateException("Invalid protobuf, no centroids were found");
        }
        DenseVector[] centroids = new DenseVector[proto.getCentroidVectorsCount()];
        List<TensorProto> centroidProtos = proto.getCentroidVectorsList();
        for (int i = 0; i < centroids.length; i++) {
            Tensor centroidTensor = Tensor.deserialize(centroidProtos.get(i));
            if (centroidTensor instanceof DenseVector) {
                DenseVector centroid = (DenseVector) centroidTensor;
                if (centroid.size() != featureDomain.size()) {
                    throw new IllegalStateException("Invalid protobuf, centroid did not contain all the features, found " + centroid.size() + " expected " + featureDomain.size());
                }
                centroids[i] = centroid;
            } else {
                throw new IllegalStateException("Invalid protobuf, expected centroid to be a dense vector, found " + centroidTensor.getClass());
            }
        }

        org.tribuo.math.distance.Distance dist = ProtoUtil.deserialize(proto.getDistance());

        return new KMeansModel(carrier.name(), carrier.provenance(), featureDomain, outputDomain, centroids, dist);
    }

    /**
     * Returns a copy of the centroids.
     * <p>
     * In most cases you should prefer {@link #getCentroids} as
     * it performs the mapping from Tribuo's internal feature ids
     * to the externally visible feature names for you.
     * This method provides direct access to the centroid vectors
     * for use in downstream processing if the ids are not relevant
     * (or are known to match).
     * @return The centroids.
     */
    public DenseVector[] getCentroidVectors() {
        DenseVector[] copies = new DenseVector[centroidVectors.length];

        for (int i = 0; i < copies.length; i++) {
            copies[i] = centroidVectors[i].copy();
        }

        return copies;
    }

    /**
     * Returns a list of features, one per centroid.
     * <p>
     * This should be used in preference to {@link #getCentroidVectors()}
     * as it performs the mapping from Tribuo's internal feature ids to
     * the externally visible feature names.
     * </p>
     * @return A list containing all the centroids.
     */
    public List<List<Feature>> getCentroids() {
        List<List<Feature>> output = new ArrayList<>(centroidVectors.length);

        for (int i = 0; i < centroidVectors.length; i++) {
            List<Feature> features = new ArrayList<>(featureIDMap.size());

            for (VectorTuple v : centroidVectors[i]) {
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
        double minDistance = Double.POSITIVE_INFINITY;
        int id = -1;
        for (int i = 0; i < centroidVectors.length; i++) {
            double distance = dist.computeDistance(centroidVectors[i], vector);

            if (distance < minDistance) {
                minDistance = distance;
                id = i;
            }
        }
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

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<ClusterID> carrier = createDataCarrier();

        KMeansModelProto.Builder modelBuilder = KMeansModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setDistance(dist.serialize());
        for (DenseVector e : centroidVectors) {
            modelBuilder.addCentroidVectors(e.serialize());
        }

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(KMeansModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected KMeansModel copy(String newName, ModelProvenance newProvenance) {
        DenseVector[] newCentroids = new DenseVector[centroidVectors.length];
        for (int i = 0; i < centroidVectors.length; i++) {
            newCentroids[i] = centroidVectors[i].copy();
        }
        return new KMeansModel(newName,newProvenance,featureIDMap,outputIDInfo,newCentroids,dist);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        if (dist == null) {
            dist = distanceType.getDistanceType().getDistance();
        }
    }
}
