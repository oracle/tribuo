/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.clustering.ClusterID;
import org.tribuo.clustering.hdbscan.HdbscanTrainer.Distance;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.ModelProvenance;

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
 */
public final class HdbscanModel extends Model<ClusterID> {
    private static final long serialVersionUID = 1L;

    private final List<Integer> clusterLabels;

    private final DenseVector outlierScoresVector;

    private final Distance distanceType;

    private final List<HdbscanTrainer.ClusterExemplar> clusterExemplars;

    HdbscanModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap,
                 ImmutableOutputInfo<ClusterID> outputIDInfo, List<Integer> clusterLabels, DenseVector outlierScoresVector,
                 List<HdbscanTrainer.ClusterExemplar> clusterExemplars, Distance distanceType) {
        super(name,description,featureIDMap,outputIDInfo,false);
        this.clusterLabels = clusterLabels;
        this.outlierScoresVector = outlierScoresVector;
        this.clusterExemplars = clusterExemplars;
        this.distanceType = distanceType;
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

    @Override
    public Prediction<ClusterID> predict(Example<ClusterID> example) {
        SparseVector vector = SparseVector.createSparseVector(example,featureIDMap,false);
        if (vector.numActiveElements() == 0) {
            throw new IllegalArgumentException("No features found in Example " + example);
        }
        double minDistance = Double.POSITIVE_INFINITY;
        int clusterLabel = -1;
        double clusterOutlierScore = 0.0;
        for (HdbscanTrainer.ClusterExemplar clusterExemplar : clusterExemplars) {
            double distance;
            switch (distanceType) {
                case EUCLIDEAN:
                    distance = clusterExemplar.getFeatures().euclideanDistance(vector);
                    break;
                case COSINE:
                    distance = clusterExemplar.getFeatures().cosineDistance(vector);
                    break;
                case L1:
                    distance = clusterExemplar.getFeatures().l1Distance(vector);
                    break;
                default:
                    throw new IllegalStateException("Unknown distance " + distanceType);
            }
            if (distance < minDistance) {
                minDistance = distance;
                clusterLabel = clusterExemplar.getLabel();
                clusterOutlierScore = clusterExemplar.getOutlierScore();
            }
        }
        return new Prediction<>(new ClusterID(clusterLabel, clusterOutlierScore),vector.size(),example);
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
    protected HdbscanModel copy(String newName, ModelProvenance newProvenance) {
        DenseVector copyOutlierScoresVector = outlierScoresVector.copy();
        List<Integer> copyClusterLabels = Collections.unmodifiableList(clusterLabels);
        List<HdbscanTrainer.ClusterExemplar> copyExemplars = new ArrayList<>(clusterExemplars);
        return new HdbscanModel(newName, newProvenance, featureIDMap, outputIDInfo, copyClusterLabels,
            copyOutlierScoresVector, copyExemplars, distanceType);
    }
}
