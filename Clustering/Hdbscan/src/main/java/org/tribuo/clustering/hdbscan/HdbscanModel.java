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

package org.tribuo.clustering.hdbscan;

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
import org.tribuo.clustering.hdbscan.HdbscanTrainer.Distance;
import org.tribuo.clustering.hdbscan.protos.ClusterExemplarProto;
import org.tribuo.clustering.hdbscan.protos.HdbscanModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.la.VectorTuple;
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
 * A trained HDBSCAN* model which provides the cluster assignment labels and outlier scores for every data point.
 * <p>
 * The predict method of this model approximates the cluster labels for new data points, based on the
 * current clustering. The model is not updated with the new data. This is a novel prediction technique which
 * leverages the computed cluster exemplars from the HDBSCAN* algorithm.
 * <p>
 * See:
 * <pre>
 * G. Stewart, M. Al-Khassaweneh. "An Implementation of the HDBSCAN* Clustering Algorithm",
 * Applied Sciences. 2022; 12(5):2405.
 * <a href="https://doi.org/10.3390/app12052405">Manuscript</a>
 * </pre>
 */
public final class HdbscanModel extends Model<ClusterID> {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final List<Integer> clusterLabels;

    private final DenseVector outlierScoresVector;

    @Deprecated
    private Distance distanceType;

    // This is not final to support deserialization of older models. It will be final in a future version which doesn't
    // maintain serialization compatibility with 4.X.
    private org.tribuo.math.distance.Distance dist;

    private final List<HdbscanTrainer.ClusterExemplar> clusterExemplars;

    private final double noisePointsOutlierScore;

    HdbscanModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap,
                 ImmutableOutputInfo<ClusterID> outputIDInfo, List<Integer> clusterLabels, DenseVector outlierScoresVector,
                 List<HdbscanTrainer.ClusterExemplar> clusterExemplars, org.tribuo.math.distance.Distance dist, double noisePointsOutlierScore) {
        super(name,description,featureIDMap,outputIDInfo,false);
        this.clusterLabels = Collections.unmodifiableList(clusterLabels);
        this.outlierScoresVector = outlierScoresVector;
        this.clusterExemplars = Collections.unmodifiableList(clusterExemplars);
        this.dist = dist;
        this.noisePointsOutlierScore = noisePointsOutlierScore;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static HdbscanModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        HdbscanModelProto proto = message.unpack(HdbscanModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(ClusterID.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not a clustering domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<ClusterID> outputDomain = (ImmutableOutputInfo<ClusterID>) carrier.outputDomain();

        Tensor outlierScoresTensor = Tensor.deserialize(proto.getOutlierScoresVector());
        if (!(outlierScoresTensor instanceof DenseVector)) {
            throw new IllegalStateException("Invalid protobuf, outlier scores must be a dense vector, found " + outlierScoresTensor.getClass());
        }
        DenseVector outlierScoresVector = (DenseVector) outlierScoresTensor;

        List<Integer> clusterLabels = new ArrayList<>(proto.getClusterLabelsList());
        for (Integer i : clusterLabels) {
            if (outputDomain.getOutput(i) == null && i != -1) {
                throw new IllegalStateException("Invalid protobuf, found cluster id " + i + " which is not present in the domain " + outputDomain);
            }
        }
        if (clusterLabels.size() != outlierScoresVector.size()) {
            throw new IllegalStateException("Invalid protobuf, expected the same number of outlier scores as cluster labels, found " +outlierScoresVector.size() + " scores and " + clusterLabels.size() + " labels");
        }

        List<HdbscanTrainer.ClusterExemplar> exemplars = new ArrayList<>();
        for (ClusterExemplarProto p : proto.getClusterExemplarsList()) {
            exemplars.add(HdbscanTrainer.ClusterExemplar.deserialize(p));
        }

        org.tribuo.math.distance.Distance dist = ProtoUtil.deserialize(proto.getDistance());

        return new HdbscanModel(carrier.name(), carrier.provenance(), carrier.featureDomain(),
            outputDomain, clusterLabels, outlierScoresVector, exemplars, dist, proto.getNoisePointsOutlierScore());
    }

    /**
     * Returns the cluster labels for the training data.
     * <p>
     * The cluster labels are in the same order as the original data points. A label of
     * {@link HdbscanTrainer#OUTLIER_NOISE_CLUSTER_LABEL} indicates an outlier or noise point.
     * @return The cluster labels for every data point from the training data.
     */
    public List<Integer> getClusterLabels() {
        return clusterLabels;
    }

    /**
     * Returns the GLOSH (Global-Local Outlier Scores from Hierarchies) outlier scores for the training data.
     * These are values between 0 and 1. A higher score indicates that a point is more likely to be an outlier.
     * <p>
     * The outlier scores are in the same order as the original data points.
     * @return The outlier scores for every data point from the training data.
     */
    public List<Double> getOutlierScores() {
        List<Double> outlierScores = new ArrayList<>(outlierScoresVector.size());
        for (double outlierScore : outlierScoresVector.toArray()) {
            outlierScores.add(outlierScore);
        }
        return outlierScores;
    }

    /**
     * Returns a deep copy of the cluster exemplars.
     * @return The cluster exemplars.
     */
    public List<HdbscanTrainer.ClusterExemplar> getClusterExemplars() {
        List<HdbscanTrainer.ClusterExemplar> list = new ArrayList<>(clusterExemplars.size());
        for (HdbscanTrainer.ClusterExemplar e : clusterExemplars) {
            list.add(e.copy());
        }
        return list;
    }

    /**
     * Returns the features in each cluster exemplar.
     * <p>
     * In many cases this should be used in preference to {@link #getClusterExemplars()}
     * as it performs the mapping from Tribuo's internal feature ids to
     * the externally visible feature names.
     * @return The cluster exemplars.
     */
    public List<Pair<Integer,List<Feature>>> getClusters() {
        List<Pair<Integer,List<Feature>>> list = new ArrayList<>(clusterExemplars.size());

        for (HdbscanTrainer.ClusterExemplar e : clusterExemplars) {
            List<Feature> features = new ArrayList<>(e.getFeatures().numActiveElements());

            for (VectorTuple v : e.getFeatures()) {
                Feature f = new Feature(featureIDMap.get(v.index).getName(),v.value);
                features.add(f);
            }

            list.add(new Pair<>(e.getLabel(),features));
        }

        return list;
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
            throw new IllegalArgumentException("No features found in Example " + example);
        }

        double minDistance = Double.POSITIVE_INFINITY;
        int clusterLabel = HdbscanTrainer.OUTLIER_NOISE_CLUSTER_LABEL;
        double outlierScore = 0.0;
        if (Double.compare(noisePointsOutlierScore, 0) > 0) { // This will be true from models > 4.2
            boolean isNoisePoint = true;
            for (HdbscanTrainer.ClusterExemplar clusterExemplar : clusterExemplars) {
                double distance = dist.computeDistance(clusterExemplar.getFeatures(), vector);
                if (isNoisePoint && distance <= clusterExemplar.getMaxDistToEdge()) {
                    isNoisePoint = false;
                }
                if (distance < minDistance) {
                    minDistance = distance;
                    clusterLabel = clusterExemplar.getLabel();
                    outlierScore = clusterExemplar.getOutlierScore();
                }
            }
            if (isNoisePoint) {
                clusterLabel = HdbscanTrainer.OUTLIER_NOISE_CLUSTER_LABEL;
                outlierScore = noisePointsOutlierScore;
            }
        }
        else {
            for (HdbscanTrainer.ClusterExemplar clusterExemplar : clusterExemplars) {
                double distance = dist.computeDistance(clusterExemplar.getFeatures(), vector);
                if (distance < minDistance) {
                    minDistance = distance;
                    clusterLabel = clusterExemplar.getLabel();
                    outlierScore = clusterExemplar.getOutlierScore();
                }
            }
        }
        return new Prediction<>(new ClusterID(clusterLabel, outlierScore),vector.size(),example);
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

        HdbscanModelProto.Builder modelBuilder = HdbscanModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.addAllClusterLabels(clusterLabels);
        modelBuilder.setOutlierScoresVector(outlierScoresVector.serialize());
        modelBuilder.setDistance(dist.serialize());
        for (HdbscanTrainer.ClusterExemplar e : clusterExemplars) {
            modelBuilder.addClusterExemplars(e.serialize());
        }
        modelBuilder.setNoisePointsOutlierScore(noisePointsOutlierScore);

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(HdbscanModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    @Override
    protected HdbscanModel copy(String newName, ModelProvenance newProvenance) {
        DenseVector copyOutlierScoresVector = outlierScoresVector.copy();
        List<Integer> copyClusterLabels = new ArrayList<>(clusterLabels);
        List<HdbscanTrainer.ClusterExemplar> copyExemplars = new ArrayList<>(clusterExemplars);
        return new HdbscanModel(newName, newProvenance, featureIDMap, outputIDInfo, copyClusterLabels,
            copyOutlierScoresVector, copyExemplars, dist, noisePointsOutlierScore);
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        if (dist == null) {
            dist = distanceType.getDistanceType().getDistance();
        }
    }
}
