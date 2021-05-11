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
 * Provides classes and infrastructure for regression problems with single or multiple output dimensions.
 * <p>
 * All Tribuo regressions are treated as if they have multiple output dimensions, there is no
 * special casing in the API for single dimensional regression. As a result the methods on
 * {@link org.tribuo.regression.Regressor} all return arrays of values. Each
 * {@link org.tribuo.regression.Regressor} can be seen as a collection of
 * {@link org.tribuo.regression.Regressor.DimensionTuple} each of which represents a single named
 * regression dimension, along with the associated regressed value and optionally a variance.
 */
package org.tribuo.regression;