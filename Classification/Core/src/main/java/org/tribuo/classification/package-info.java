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
 * Provides classes and infrastructure for multiclass classification problems.
 * <p>
 * The multiclass {@link org.tribuo.Output} implementation is called
 * {@link org.tribuo.classification.Label} and has an associated
 * {@link org.tribuo.evaluation.Evaluator}, {@link org.tribuo.OutputFactory}
 * and {@link org.tribuo.OutputInfo}s.
 * </p>
 * <p>
 * Note: Tribuo has no specialisation for binary classification, all classification tasks are considered to be
 * multiclass, and all classification models work for multiclass problems.
 * </p>
 */
package org.tribuo.classification;