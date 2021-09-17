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

package org.tribuo.math.optimisers.util;

import org.tribuo.math.la.Tensor;

/**
 * An interface which tags a {@link Tensor} with a convertToDense method.
 */
public interface ShrinkingTensor {
    /**
     * The tolerance below which the scale factor is applied to the stored values and reset to 1.0.
     */
    public static final double tolerance = 1e-6;

    /**
     * Converts the tensor into a dense tensor.
     * @return A dense tensor copy of this shrinking tensor.
     */
    public Tensor convertToDense();
}


