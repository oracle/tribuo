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

package org.tribuo.util;

import java.util.Comparator;
import java.util.Objects;

/**
 * A Pair of a primitive int and a primitive double.
 */
public final class IntDoublePair {

    /**
     * The key.
     */
    public final int index;

    /**
     * The value.
     */
    public final double value;

    /**
     * Constructs a tuple out of an int and a double.
     * @param index The int.
     * @param value The double
     */
    public IntDoublePair(int index, double value) {
        this.index = index;
        this.value = value;
    }

    /**
     * Compare pairs by index.
     * @return Comparator over indices.
     */
    public static Comparator<IntDoublePair> pairIndexComparator() {
        return Comparator.comparingInt(a -> a.index);
    }

    /**
     * Compare pairs by value. Ascending order.
     * @return Comparator over absolute values.
     */
    public static Comparator<IntDoublePair> pairValueComparator() {
        return Comparator.comparingDouble(a -> Math.abs(a.value));
    }

    /**
     * Compare pairs by value. Descending order.
     * @return Comparator over absolute values.
     */
    public static Comparator<IntDoublePair> pairDescendingValueComparator() {
        return (IntDoublePair a, IntDoublePair b) -> Double.compare(Math.abs(b.value), Math.abs(a.value));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IntDoublePair that = (IntDoublePair) o;
        return index == that.index &&
                Double.compare(that.value, value) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, value);
    }

    @Override
    public String toString() {
        return "(" + index + "," + value + ")";
    }
}
