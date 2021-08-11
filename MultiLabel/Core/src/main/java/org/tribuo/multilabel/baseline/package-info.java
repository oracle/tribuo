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
 * Provides implementations of binary relevance based multi-label classification
 * algorithms.
 * <p>
 * {@link org.tribuo.multilabel.baseline.IndependentMultiLabelTrainer} provides
 * the standard binary relevance algorithm where n models are built independently,
 * one per label in the domain.
 * <p>
 * {@link org.tribuo.multilabel.baseline.ClassifierChainTrainer} provides
 * classifier chains, which train n models in a sequence, one per label, where
 * each model observes the labels before it in the chain. This label ordering
 * can be random, or specified on construction.
 */
package org.tribuo.multilabel.baseline;