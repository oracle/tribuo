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

import com.oracle.labs.mlrg.olcut.util.Pair;

import java.util.ArrayList;

/**
 * A pair of things with a cached hashcode.
 * <p>
 * The cache is calculated on construction, and the objects inside the pair are thus expected to be immutable.
 * If they aren't then the behaviour is undefined (and you shouldn't use this class).
 * @param <T1> The type of the first object.
 * @param <T2> The type of the second object.
 */
public class CachedPair<T1, T2> extends Pair<T1,T2> {
    private static final long serialVersionUID = 1L;

    private final int cachedHash;

    /**
     * Constructs a CachedPair.
     * @param a The first element.
     * @param b The second element.
     */
    public CachedPair(T1 a, T2 b) {
        super(a,b);
        this.cachedHash = super.hashCode();
    }

    /**
     * Takes two arrays and zips them together into an array of CachedPairs.
     * @param <T1> The type contained in the first array.
     * @param <T2> The type contained in the second array.
     * @param first An array of values.
     * @param second Another array of values.
     * @return The zipped array.
     */
    public static <T1,T2> ArrayList<CachedPair<T1,T2>> zipArraysCached(ArrayList<T1> first, ArrayList<T2> second) {
        if (first.size() == second.size()) {
            ArrayList<CachedPair<T1,T2>> output = new ArrayList<>(first.size());

            for (int i = 0; i < first.size(); i++) {
                CachedPair<T1,T2> pair = new CachedPair<>(first.get(i),second.get(i));
                output.add(i, pair);
            }

            return output;
        } else {
            throw new IllegalArgumentException("Zipping requires arrays of the same length. first.size() = " + first.size() + ", second.size() = " + second.size());
        }
    }

    /**
     * Overridden hashcode.
     * Uses the cached value calculated on construction.
     * @return A 32-bit integer.
     */
    @Override
    public int hashCode() {
        return cachedHash;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        CachedPair<?, ?> that = (CachedPair<?, ?>) o;
        return cachedHash == that.cachedHash;
    }
}
