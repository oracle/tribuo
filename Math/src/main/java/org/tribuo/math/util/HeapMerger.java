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

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.la.VectorIterator;
import org.tribuo.math.la.VectorTuple;
import org.tribuo.math.protos.MergerProto;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.logging.Logger;

/**
 * Merges each {@link SparseVector} separately using a {@link PriorityQueue} as a heap.
 * <p>
 * Relies upon {@link VectorIterator#compareTo(VectorIterator)}.
 */
public class HeapMerger implements Merger {
    private static final long serialVersionUID = 1L;
    private static final Logger logger = Logger.getLogger(HeapMerger.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a HeapMerger.
     */
    public HeapMerger() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static HeapMerger deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new HeapMerger();
    }

    @Override
    public MergerProto serialize() {
        MergerProto.Builder mergerProto = MergerProto.newBuilder();
        mergerProto.setClassName(this.getClass().getName());
        mergerProto.setVersion(CURRENT_VERSION);
        return mergerProto.build();
    }

    @Override
    public DenseSparseMatrix merge(DenseSparseMatrix[] inputs) {
        int denseLength = inputs[0].getDimension1Size();
        int sparseLength = inputs[0].getDimension2Size();
        int[] totalLengths = new int[inputs[0].getDimension1Size()];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < totalLengths.length; j++) {
                totalLengths[j] += inputs[i].numActiveElements(j);
            }
        }

        int maxLength = 0;
        for (int i = 0; i < totalLengths.length; i++) {
            if (totalLengths[i] > maxLength) {
                maxLength = totalLengths[i];
            }
        }

        SparseVector[] output = new SparseVector[denseLength];

        int[] indicesBuffer = new int[maxLength];
        double[] valuesBuffer = new double[maxLength];
        
        List<SparseVector> vectors = new ArrayList<>();
        for (int i = 0; i < denseLength; i++) {
            vectors.clear();
            for (DenseSparseMatrix m : inputs) {
                SparseVector vec = m.getRow(i);
                if (vec.numActiveElements() > 0) {
                    vectors.add(vec);
                }
            }
            output[i] = merge(vectors,sparseLength,indicesBuffer,valuesBuffer);
        }

        return DenseSparseMatrix.createFromSparseVectors(output);
    }

    @Override
    public SparseVector merge(SparseVector[] inputs) {
        int maxLength = 0;

        for (int i = 0; i < inputs.length; i++) {
            maxLength += inputs[i].numActiveElements();
        }

        return merge(Arrays.asList(inputs),inputs[0].size(),new int[maxLength],new double[maxLength]);
    }

    /**
     * Merges a list of sparse vectors into a single sparse vector, summing the values.
     * @param vectors The vectors to merge.
     * @param dimension The dimension of the sparse vector.
     * @param indicesBuffer A buffer for the indices.
     * @param valuesBuffer A buffer for the values.
     * @return The merged SparseVector.
     */
    public static SparseVector merge(List<SparseVector> vectors, int dimension, int[] indicesBuffer, double[] valuesBuffer) {
        PriorityQueue<VectorIterator> queue = new PriorityQueue<>();
        Arrays.fill(valuesBuffer,0.0);

        for (SparseVector vector : vectors) {
            // Setup matrix iterators, call next to load the first value.
            VectorIterator cur = vector.iterator();
            cur.next();
            queue.add(cur);
        }

        int sparseCounter = 0;
        int sparseIndex = -1;

        while (!queue.isEmpty()) {
            VectorIterator cur = queue.peek();
            VectorTuple ref = cur.getReference();
            //logger.log(Level.INFO,"Tuple=" + ref.toString() + ", itrName="+((DenseSparseMatrix.DenseSparseMatrixIterator)cur).getName()+", sparseIndex="+sparseIndex+", sparseCounter="+sparseCounter+", denseCounter="+denseCounter);
            //logger.log(Level.INFO,"Queue=" + queue.toString());

            if (sparseIndex == -1) {
                //if we're at the start, store the first value
                sparseIndex = ref.index;
                indicesBuffer[sparseCounter] = sparseIndex;
                valuesBuffer[sparseCounter] = ref.value;
            } else if (ref.index == sparseIndex) {
                //if we're already in the right place, aggregate value
                valuesBuffer[sparseCounter] += ref.value;
            } else {
                //else increment the sparseCounter and store the new value
                sparseIndex = ref.index;
                sparseCounter++;
                indicesBuffer[sparseCounter] = sparseIndex;
                valuesBuffer[sparseCounter] = ref.value;
            }

            if (!cur.hasNext()) {
                //Discard exhausted iterator
                queue.poll();
            } else {
                //consume the value and reheap
                cur.next();
                VectorIterator tmp = queue.poll();
                queue.offer(tmp);
            }
        }
        //Generate the final SparseVector
        int[] indices = Arrays.copyOf(indicesBuffer,sparseCounter+1);
        double[] values = Arrays.copyOf(valuesBuffer,sparseCounter+1);
        return SparseVector.createSparseVector(dimension,indices,values);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (o == null || getClass() != o.getClass()) {
            return false;
        } else {
            return true;
        }
    }

    @Override
    public int hashCode() {
        return 31;
    }
}
