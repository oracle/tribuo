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
 * Provides implementations of {@link org.tribuo.data.columnar.FieldExtractor}.
 *
 * <P>
 *
 * The {@link org.tribuo.data.columnar.FieldExtractor#getMetadataName()} values of "name" {@link org.tribuo.Example#NAME}
 * and "weight" (case-sensitive) are special. In particular, "name" is used throughout the {@link org.tribuo.Example}
 * environment to uniquely identify Examples within a {@link org.tribuo.Dataset}. "weight" should only
 * be used for the {@link org.tribuo.data.columnar.FieldExtractor} supplied to {@link org.tribuo.data.columnar.RowProcessor}
 * as the {@code weightExtractor} which is used by the system to weight {@link org.tribuo.Example}s.
 */
package org.tribuo.data.columnar.extractors;