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
 * Provides the base interface and implementations of the {@link org.tribuo.Model} hashing
 * which obscures the feature names stored in a model.
 * <p>
 * The base interface is {@link org.tribuo.hash.Hasher}, which has three implementations:
 * {@link org.tribuo.hash.HashCodeHasher} which uses String.hashCode(),
 * {@link org.tribuo.hash.ModHashCodeHasher} which uses String.hashCode() and remaps the output into
 * a specific range, and {@link org.tribuo.hash.MessageDigestHasher} which uses a
 * {@link java.security.MessageDigest} implementation to perform the hashing. Only MessageDigestHasher provides
 * security guarantees suitable for production usage.
 * <p>
 * Note {@link org.tribuo.hash.Hasher} implementations require a salt which is serialized separately
 * from the {@code Hasher} object. Without supplying this seed the hash methods will throw
 * {@link java.lang.IllegalStateException}.
 */
package org.tribuo.hash;