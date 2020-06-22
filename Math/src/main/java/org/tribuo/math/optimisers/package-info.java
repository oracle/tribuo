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
 * Provides implementations of {@link org.tribuo.math.StochasticGradientOptimiser}.
 * <p>
 * Has implementations of SGD using a variety of simple learning rate tempering systems, along with AdaGrad, Adam,
 * AdaDelta, RMSProp and Pegasos.
 * </p>
 * <p>
 * Also provides {@link org.tribuo.math.optimisers.ParameterAveraging}
 * which wraps another {@link org.tribuo.math.StochasticGradientOptimiser} and averages the
 * learned parameters across the gradient descent run. This is usually used for convex problems, for non-convex ones
 * your milage may vary.
 * </p>
 */
package org.tribuo.math.optimisers;