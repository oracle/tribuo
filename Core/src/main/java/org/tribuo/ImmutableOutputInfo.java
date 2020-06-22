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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.util.Pair;

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

}
