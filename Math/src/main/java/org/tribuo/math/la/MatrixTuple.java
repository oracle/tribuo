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

import java.util.Objects;

/**
 * A mutable tuple used to avoid allocation when iterating a matrix.
 * <p>
 * While it does implement hashcode, don't use it as a key as the hashcode is data dependent.
 */
public class MatrixTuple {

    /**
     * The i index.
     */
    public int i;
    /**
     * The j index.
     */
    public int j;

    /**
     * The value.
     */
    public double value;

    /**
     * Constructs an empty matrix tuple.
     */
    public MatrixTuple() {
        this.i = -1;
        this.j = -1;
        this.value = Double.NaN;
    }

    /**
     * Constructs a matrix tuple with the specified values.
     * @param i The i index.
     * @param j The j index.
     * @param value The value.
     */
    public MatrixTuple(int i, int j, int value) {
        this.i = i;
        this.j = j;
        this.value = value;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof MatrixTuple) {
            MatrixTuple otherM = (MatrixTuple) o;
            if ((i == otherM.i) && (j == otherM.j)) {
                return Math.abs(value - otherM.value) < 1e-12;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hash(i, j, value);
    }

    @Override
    public String toString() {
        return "MatrixTuple(i="+i+",j="+j+",value="+value+")";
    }
}
