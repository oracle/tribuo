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

package org.tribuo.clustering.evaluation;

import org.tribuo.clustering.ClusterID;
import org.tribuo.evaluation.Evaluation;

/**
 * An {@link Evaluation} for clustering tasks.
 */
public interface ClusteringEvaluation extends Evaluation<ClusterID> {

    /**
     * Calculates the normalized MI between the ground truth clustering ids and the predicted ones.
     * <p>
     * The value is bounded between 0 and 1.
     * <p>
     * If this value is 1, then the predicted id values are a permutation of the supplied ids.
     * If the value is 0 then the predicted ids are random wrt the supplied ids.
     * @return The normalized MI.
     */
    double normalizedMI();

    /**
     * Measures the adjusted normalized mutual information between the predicted ids and the supplied ids.
     * <p>
     * The value is bounded between 0 and 1.
     * <p>
     * If this value is 1, then the predicted id values are a permutation of the supplied ids.
     * If the value is 0 then the predicted ids are random wrt the supplied ids.
     * <p>
     * It's adjusted for chance unlike the normalized one.
     * @return The adjusted MI.
     */
    double adjustedMI();

}