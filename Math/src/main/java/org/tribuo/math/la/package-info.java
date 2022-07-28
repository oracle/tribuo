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
 * Provides a linear algebra system used for numerical operations in Tribuo.
 * <p>
 * There are Dense and Sparse vectors and Matrices, along with a DenseSparseMatrix which is
 * a dense array of sparse row vectors. The dense matrix provides various factorization methods
 * in addition to matrix-vector operations.
 * <p>
 * It's a single threaded implementation in pure Java. We're looking at ways of improving the speed
 * using new technologies coming in future releases of Java.
 */
package org.tribuo.math.la;