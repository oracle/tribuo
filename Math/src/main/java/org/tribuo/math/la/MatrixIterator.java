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

package org.tribuo.math.la;

import java.util.Iterator;

/**
 * A {@link Comparable} {@link Iterator} over {@link MatrixTuple}s.
 * <p>
 * The comparison is based on the indices, not on the values.
 * To use the comparator .next must have been called once.
 */
public interface MatrixIterator extends Iterator<MatrixTuple>, Comparable<MatrixIterator> {

    /**
     * Gets the MatrixTuple reference that this iterator updates.
     * @return The matrix tuple.
     */
    public MatrixTuple getReference();

    @Override
    public default int compareTo(MatrixIterator o) {
        MatrixTuple ours = getReference();
        MatrixTuple other = o.getReference();
        if (other.i < ours.i) {
            return 1;
        } else if (other.i > ours.i) {
            return -1;
        } else {
            //other.i == pair.i;
            return Integer.compare(ours.j, other.j);
        }
    }

}
