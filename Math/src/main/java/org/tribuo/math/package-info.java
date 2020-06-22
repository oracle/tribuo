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

/**
 * Contains the implementation of Tribuo's math library, it's gradient descent optimisers, kernels and a set of
 * math related utils.
 * <p>
 * This package contains two core interfaces for Tribuo's gradient descent system,
 * {@link org.tribuo.math.Parameters} which controls how parameters updated via gradient descent are
 * accessed, and {@link org.tribuo.math.StochasticGradientOptimiser} which provides the optimisation
 * interface.
 */
package org.tribuo.math;