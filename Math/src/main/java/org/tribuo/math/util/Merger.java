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

package org.tribuo.math.util;

import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.protos.MergerProto;
import org.tribuo.protos.ProtoSerializable;

import java.io.Serializable;

/**
 * An interface for merging an array of {@link DenseSparseMatrix} into a single {@link DenseSparseMatrix}.
 * <p>
 * Mergers are principally used to aggregate gradients across a minibatch.
 * <p>
 * Merging is done by summation.
 */
public interface Merger extends ProtoSerializable<MergerProto>, Serializable {

    /**
     * Merges an array of DenseSparseMatrix into a single DenseSparseMatrix.
     * @param inputs The matrices to merge.
     * @return The merged matrix.
     */
    public DenseSparseMatrix merge(DenseSparseMatrix[] inputs);

    /**
     * Merges an array of SparseVector into a single SparseVector.
     * @param inputs The vectors to merge.
     * @return The merged vector.
     */
    public SparseVector merge(SparseVector[] inputs);

}
