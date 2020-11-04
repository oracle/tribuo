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
 * Provides math related util classes.
 * <p>
 * There are two interfaces: {@link org.tribuo.math.util.Merger} and
 * {@link org.tribuo.math.util.VectorNormalizer}. Merger is used to combine gradient tensors
 * produced across different examples (i.e., to combine the gradients within a minibatch). VectorNormalizer is
 * used to normalize an array, usually into a probability distribution.
 */
package org.tribuo.math.util;