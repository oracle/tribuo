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

package org.tribuo.util.infotheory.impl;

/**
 * A mutable tuple of a double and a long.
 */
public final class WeightCountTuple {

    /**
     * The current weight.
     */
    public double weight;
    /**
     * The current count.
     */
    public long count;

    /**
     * Creates a zeroed WeightCountTuple.
     */
    public WeightCountTuple() {
        this.weight = 0.0;
        this.count = 0;
    }

    /**
     * Creates a WeightCountTuple with the specifed values.
     * @param weight The current weight.
     * @param count The current count.
     */
    public WeightCountTuple(double weight, long count) {
        this.weight = weight;
        this.count = count;
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 47 * hash + (int) (Double.doubleToLongBits(this.weight) ^ (Double.doubleToLongBits(this.weight) >>> 32));
        hash = 47 * hash + (int) (this.count ^ (this.count >>> 32));
        return hash;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (!(obj instanceof WeightCountTuple)) {
            return false;
        }
        final WeightCountTuple other = (WeightCountTuple) obj;
        return (this.weight == other.weight) && (this.count == other.count);
    }
    
}
