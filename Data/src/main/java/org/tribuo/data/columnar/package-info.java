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
 * Provides classes for processing columnar data and generating {@link org.tribuo.Example}s.
 * <p>
 * The main class is {@link org.tribuo.data.columnar.RowProcessor} which can take a {@link java.util.Map}
 * from String to String and generate {@link org.tribuo.Feature}s using a {@link org.tribuo.data.columnar.FieldProcessor}
 * and {@link org.tribuo.Output}s using a {@link org.tribuo.data.columnar.ResponseProcessor}.
 */
package org.tribuo.data.columnar;