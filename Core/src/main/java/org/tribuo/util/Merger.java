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

/**
 * An interface which can merge double values.
 * <p>
 * Provides a few obvious stateless examples.
 */
@FunctionalInterface
public interface Merger {

    /**
     * Merges first and second.
     * @param first The first input.
     * @param second The second input.
     * @return The merged value.
     */
    public double merge(double first, double second);

    /**
     * A merger which takes the maximum element.
     * @return The maximum function.
     */
    public static Merger max() {
        return Math::max;
    }

    /**
     * A merger which takes the minimum element.
     * @return The minimum function.
     */
    public static Merger min() {
        return Math::min;
    }

    /**
     * A merger which adds the elements.
     * @return The addition function.
     */
    public static Merger add() {
        return Double::sum;
    }

}
