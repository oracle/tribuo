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

package org.tribuo.clustering.kmeans;

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
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.provenance.ModelProvenance;

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

    private final DenseVector[] centroidVectors;

    private final Distance distanceType;

    KMeansModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<ClusterID> outputIDInfo, DenseVector[] centroidVectors, Distance distanceType) {
        super(name,description,featureIDMap,outputIDInfo,false);
        this.centroidVectors = centroidVectors;
        this.distanceType = distanceType;
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
            double distance;
            switch (distanceType) {
                case EUCLIDEAN:
                    distance = centroidVectors[i].euclideanDistance(vector);
                    break;
                case COSINE:
                    distance = centroidVectors[i].cosineDistance(vector);
                    break;
                case L1:
                    distance = centroidVectors[i].l1Distance(vector);
                    break;
                default:
                    throw new IllegalStateException("Unknown distance " + distanceType);
            }
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
    protected KMeansModel copy(String newName, ModelProvenance newProvenance) {
        DenseVector[] newCentroids = new DenseVector[centroidVectors.length];
        for (int i = 0; i < centroidVectors.length; i++) {
            newCentroids[i] = centroidVectors[i].copy();
        }
        return new KMeansModel(newName,newProvenance,featureIDMap,outputIDInfo,newCentroids,distanceType);
    }
}
