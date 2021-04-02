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

package org.tribuo.transform;

/**
 * An interface for the statistics that need to be
 * collected for a specific {@link Transformation} on
 * a single feature.
 * <p>
 * TransformStatistics are purely a runtime collection,
 * they should not be persisted, nor can be directly
 * constructed.
 */
public interface TransformStatistics {

    /**
     * Observes a value and updates the statistics.
     * @param value The value to observe.
     */
    public void observeValue(double value);

    /**
     * Observes a sparse (i.e., zero) value.
     * @deprecated in 4.1 as it's unnecessary.
     */
    @Deprecated
    public void observeSparse();

    /**
     * Observes {@code count} sparse values.
     * @param count The number of sparse values.
     */
    public void observeSparse(int count);

    /**
     * Generates the appropriate {@link Transformer}
     * from the collected statistics.
     * @return The Transformer.
     */
    public Transformer generateTransformer();

}
