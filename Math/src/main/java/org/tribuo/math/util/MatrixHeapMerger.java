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
import org.tribuo.math.la.MatrixIterator;
import org.tribuo.math.la.MatrixTuple;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.protos.MergerProto;

import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.logging.Logger;

/**
 * Merges each {@link DenseSparseMatrix} using a {@link PriorityQueue} as a heap on the {@link MatrixIterator}.
 * <p>
 * Relies upon {@link MatrixIterator#compareTo(MatrixIterator)}.
 */
public class MatrixHeapMerger implements Merger {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(MatrixHeapMerger.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    /**
     * Constructs a HeapMerger.
     */
    public MatrixHeapMerger() {}

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @return The deserialized object.
     */
    public static MatrixHeapMerger deserializeFromProto(int version, String className, Any message) {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (message.getValue() != ByteString.EMPTY) {
            throw new IllegalArgumentException("Invalid proto");
        }
        return new MatrixHeapMerger();
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
        int sparseLength = inputs[0].getDimension2Size();
        PriorityQueue<MatrixIterator> queue = new PriorityQueue<>();
        int[] totalLengths = new int[inputs[0].getDimension1Size()];

        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < totalLengths.length; j++) {
                totalLengths[j] += inputs[i].numActiveElements(j);
            }
            // Setup matrix iterators, call next to load the first value.
            MatrixIterator cur = inputs[i].iterator();
            cur.next();
            queue.add(cur);
        }

        int maxLength = 0;
        for (int i = 0; i < totalLengths.length; i++) {
            if (totalLengths[i] > maxLength) {
                maxLength = totalLengths[i];
            }
        }

        SparseVector[] output = new SparseVector[totalLengths.length];

        int denseCounter = 0;
        int sparseCounter = 0;
        int sparseIndex = -1;

        int[] curIndices = new int[maxLength];
        double[] curValues = new double[maxLength];

        while (!queue.isEmpty()) {
            MatrixIterator cur = queue.peek();
            MatrixTuple ref = cur.getReference();
            //logger.log(Level.INFO,"Tuple=" + ref.toString() + ", itrName="+((DenseSparseMatrix.DenseSparseMatrixIterator)cur).getName()+", sparseIndex="+sparseIndex+", sparseCounter="+sparseCounter+", denseCounter="+denseCounter);
            //logger.log(Level.INFO,"Queue=" + queue.toString());
            if (ref.i > denseCounter) {
                //Reached the end of the current SparseVector, so generate it and reset
                int[] indices = Arrays.copyOf(curIndices,sparseCounter+1);
                double[] values = Arrays.copyOf(curValues,sparseCounter+1);
                output[denseCounter] = SparseVector.createSparseVector(sparseLength,indices,values);
                Arrays.fill(curIndices,0);
                Arrays.fill(curValues,0);
                sparseIndex = -1;
                sparseCounter = 0;
                denseCounter++;
            }

            if (sparseIndex == -1) {
                //if we're at the start, store the first value
                sparseIndex = ref.j;
                curIndices[sparseCounter] = sparseIndex;
                curValues[sparseCounter] = ref.value;
            } else if (ref.j == sparseIndex) {
                //if we're already in the right place, aggregate value
                curValues[sparseCounter] += ref.value;
            } else {
                //else increment the sparseCounter and store the new value
                sparseIndex = ref.j;
                sparseCounter++;
                curIndices[sparseCounter] = sparseIndex;
                curValues[sparseCounter] = ref.value;
            }

            if (!cur.hasNext()) {
                //Discard exhausted iterator
                queue.poll();
            } else {
                //consume the value and reheap
                cur.next();
                MatrixIterator tmp = queue.poll();
                queue.offer(tmp);
            }
        }
        //Generate the final SparseVector
        int[] indices = Arrays.copyOf(curIndices,sparseCounter+1);
        double[] values = Arrays.copyOf(curValues,sparseCounter+1);
        output[denseCounter] = SparseVector.createSparseVector(sparseLength,indices,values);

        return DenseSparseMatrix.createFromSparseVectors(output);
    }

    @Override
    public SparseVector merge(SparseVector[] inputs) {
        int maxLength = 0;

        for (int i = 0; i < inputs.length; i++) {
            maxLength += inputs[i].numActiveElements();
        }

        return HeapMerger.merge(Arrays.asList(inputs),inputs[0].size(),new int[maxLength],new double[maxLength]);
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
