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
 * Provides classes which can load columnar data (using a {@link org.tribuo.data.columnar.RowProcessor})
 * from a SQL source.
 * <p>
 * N.B. The classes in this package accept raw SQL strings and execute them directly via JDBC. They DO NOT perform
 * any SQL escaping or other injection prevention. It is the user's responsibility to ensure that SQL passed to these
 * classes performs as desired.
 */
package org.tribuo.data.sql;
