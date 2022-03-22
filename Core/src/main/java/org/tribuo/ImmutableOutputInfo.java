/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.util.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * An {@link OutputInfo} that is fixed, and contains an id number for each valid output.
 * <p>
 * In the case of real valued outputs, the id number represents the dimension.
 */
public interface ImmutableOutputInfo<T extends Output<T>> extends OutputInfo<T>, Iterable<Pair<Integer,T>> {

    /**
     * Return the id number associated with this output, or -1 if the output is unknown.
     * @param output An output
     * @return A non-negative integer if the output is known, or -1 if the output is unknown.
     */
    public int getID(T output);

    /**
     * Returns the output associated with this id, or null if the id is unknown.
     * @param id A non-negative integer.
     * @return An output object or null if it's unknown.
     */
    public T getOutput(int id);

    /**
     * Returns the total number of observed outputs seen by this ImmutableOutputInfo.
     * @return The number of observed outputs.
     */
    public long getTotalObservations();

    /**
     * Checks if the domain is the same as the other output info's domain, and that
     * each element is mapped to the same id number.
     * <p>
     * Note the default behaviour will be removed in a future major release, and should be
     * overridden for performance reasons in all implementing classes.
     * @param other The output info to compare.
     * @return True if the domains and ids are the same.
     */
    default public boolean domainAndIDEquals(ImmutableOutputInfo<T> other) {
        List<Pair<Integer,T>> self = new ArrayList<>();
        for (Pair<Integer, T> p : this) {
            self.add(p);
        }
        self.sort(Comparator.comparingInt(Pair::getA));

        List<Pair<Integer,T>> otherList = new ArrayList<>();
        for (Pair<Integer, T> p : other) {
            otherList.add(p);
        }
        otherList.sort(Comparator.comparingInt(Pair::getA));

        return self.equals(otherList);
    }

}
