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
 * from a CSV (or other character delimited format) file.
 * <p>
 * {@link org.tribuo.data.csv.CSVDataSource} is the main way of loading CSV format data into Tribuo.
 * It provides full control over featurisation, output processing and metadata extraction.
 * {@link org.tribuo.data.csv.CSVLoader} is for simple numerical CSV files where all the
 * non-response columns should be treated as features. {@link org.tribuo.data.csv.CSVSaver} writes
 * out a Tribuo {@link org.tribuo.Dataset} in CSV format suitable for loading via
 * {@link org.tribuo.data.csv.CSVLoader} or some external tool
 */
package org.tribuo.data.csv;