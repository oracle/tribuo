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

package org.tribuo.regression.rtree.impurity;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import org.tribuo.common.tree.impl.IntArrayContainer;

import java.util.List;

/**
 * Calculates a tree impurity score based on the regression targets.
 */
public interface RegressorImpurity extends Configurable, Provenancable<ConfiguredObjectProvenance> {

    /**
     * Calculates the impurity based on the supplied weights and targets.
     * @param targets The targets.
     * @param weights The weights.
     * @return The impurity.
     */
    public double impurity(float[] targets, float[] weights);

    /**
     * Calculates the weighted impurity of the targets specified in the indices array.
     * @param indices The indices in the targets and weights arrays.
     * @param indicesLength The number of values to use in indices.
     * @param targets The regression targets.
     * @param weights The example weights.
     * @return A tuple containing the impurity and the used weight sum.
     */
    public ImpurityTuple impurityTuple(int[] indices, int indicesLength, float[] targets, float[] weights);

    /**
     * Calculates the weighted impurity of the targets specified in all the indices arrays.
     * @param indices The indices in the targets and weights arrays.
     * @param targets The regression targets.
     * @param weights The example weights.
     * @return A tuple containing the impurity and the used weight sum.
     */
    public ImpurityTuple impurityTuple(List<int[]> indices, float[] targets, float[] weights);

    /**
     * Calculates the weighted impurity of the targets specified in the indices array.
     * @param indices The indices in the targets and weights arrays.
     * @param indicesLength The number of values to use in indices.
     * @param targets The regression targets.
     * @param weights The example weights.
     * @return The weighted impurity.
     */
    default public double impurity(int[] indices, int indicesLength, float[] targets, float[] weights) {
        return impurityTuple(indices, indicesLength, targets, weights).impurity;
    }

    /**
     * Calculates the weighted impurity of the targets specified in all the indices arrays.
     * @param indices The indices in the targets and weights arrays.
     * @param targets The regression targets.
     * @param weights The example weights.
     * @return The weighted impurity.
     */
    default public double impurity(List<int[]> indices, float[] targets, float[] weights) {
        return impurityTuple(indices,targets,weights).impurity;
    }

    /**
     * Calculates the weighted impurity of the targets specified in the indices array.
     * @param indices The indices in the targets and weights arrays.
     * @param targets The regression targets.
     * @param weights The example weights.
     * @return The weighted impurity.
     */
    default public double impurity(int[] indices, float[] targets, float[] weights) {
        return impurity(indices, indices.length, targets, weights);
    }

    /**
     * Calculates the weighted impurity of the targets specified in the indices container.
     * @param indices The indices in the targets and weights arrays.
     * @param targets The regression targets.
     * @param weights The example weights.
     * @return The weighted impurity.
     */
    default public double impurity(IntArrayContainer indices, float[] targets, float[] weights) {
        return impurity(indices.array, indices.size, targets, weights);
    }

    /**
     * Tuple class for the impurity and summed weight. Will be a record one day.
     */
    public static class ImpurityTuple {
        /**
         * The impurity value.
         */
        public final float impurity;
        /**
         * The sum of the weights.
         */
        public final float weight;

        /**
         * Construct an impurity tuple.
         * @param impurity The impurity value.
         * @param weight The sum of the weights.
         */
        public ImpurityTuple(float impurity, float weight) {
            this.impurity = impurity;
            this.weight = weight;
        }
    }
}
