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

package org.tribuo.math;

import org.tribuo.math.la.Tensor;

import java.io.Serializable;

/**
 * An interface to a {@link Tensor}[] array which accepts updates to the parameters.
 * <p>
 * Parameters is essentially an SGD model at training time.
 * <p>
 * Subclasses of this can add methods for calculating gradients for
 * their prediction task, or use an external objective class.
 * <p>
 * Implementations must be serializable.
 */
public interface Parameters extends Serializable {

    /**
     * Generates an empty copy of the underlying {@link Tensor} array.
     * <p>
     * It's the same size and shape as the parameters, but all the values are 0.0.
     * @return A copy of the parameters where all values are 0.0.
     */
    public Tensor[] getEmptyCopy();

    /**
     * Get a reference to the underlying {@link Tensor} array.
     * @return The parameters.
     */
    public Tensor[] get();

    /**
     * Set the underlying {@link Tensor} array to newWeights.
     * @param newWeights New parameters to store in this object.
     */
    public void set(Tensor[] newWeights);

    /**
     * Apply gradients to the parameters. Assumes that gradients is the same length as the parameters,
     * and each {@link Tensor} is the same size as the corresponding one from the parameters.
     * <p>
     * The gradients are added to the parameters.
     * @param gradients A {@link Tensor} array of updates, with the length equal to {@link Parameters#get()}.length.
     */
    public void update(Tensor[] gradients);

    /**
     * Merge together an array of gradient arrays. Assumes the first dimension
     * is the number of gradient arrays and the second dimension is the
     * number of parameter {@link Tensor}s.
     * <p>
     * For performance reasons this call may mutate the input gradient array, and
     * may return a subset of those elements as the merge output.
     * @param gradients An array of gradient update arrays.
     * @param size The number of elements of gradients to merge. Allows gradients to have unused elements.
     * @return A single {@link Tensor} array of the summed gradients.
     */
    public Tensor[] merge(Tensor[][] gradients, int size);

}
