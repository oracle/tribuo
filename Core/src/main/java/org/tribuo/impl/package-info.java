/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
 * Provides implementations of base classes and interfaces from {@link org.tribuo}.
 * <p>
 * It contains two base implementations of {@link org.tribuo.Example}, one using
 * primitive arrays and one using a {@link java.util.List} of {@link org.tribuo.Feature} objects.
 * The array implementation should be preferred for performance reasons until Feature becomes a value type. There is
 * a subclass of {@link org.tribuo.impl.ArrayExample} which stores the feature indices. This is a performance
 * optimisation and should not be used without proper consideration.
 */
package org.tribuo.impl;