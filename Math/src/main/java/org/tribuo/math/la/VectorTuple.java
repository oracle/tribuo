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

/**
 * A mutable tuple used to avoid allocation when iterating a vector.
 * <p>
 * It's basically a cursor into a vector.
 * <p>
 * While it does implement hashcode, don't use it as a key as the hashcode is data dependent.
 */
public class VectorTuple {

    /**
     * The tolerance for equality in value comparisons.
     */
    public static final double DELTA = 1e-12;

    /**
     * The current index.
     */
    public int index;

    /**
     * The current value.
     */
    public double value;

    /**
     * Creates an empty vector tuple.
     */
    public VectorTuple() {
        this.index = -1;
        this.value = Double.NaN;
    }

    /**
     * Creates a vector tuple with the specified values.
     * @param index The current index.
     * @param value The current value.
     */
    public VectorTuple(int index, int value) {
        this.index = index;
        this.value = value;
    }

    @Override
    public boolean equals(Object o) {
        if (o instanceof VectorTuple) {
            VectorTuple otherM = (VectorTuple) o;
            if (index == otherM.index) {
                return (Math.abs(value - otherM.value) < DELTA);
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int result = index;
        result = 31 * result + (int) Double.doubleToLongBits(value);
        return result;
    }

    @Override
    public String toString() {
        return "VectorTuple(index="+index+",value="+value+")";
    }
}
