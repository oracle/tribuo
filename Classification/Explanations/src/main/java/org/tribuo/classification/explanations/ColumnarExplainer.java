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

package org.tribuo.classification.explanations;

import org.tribuo.Example;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.data.columnar.RowProcessor;

import java.util.Map;

/**
 * An explainer for data using Tribuo's columnar data package.
 */
public interface ColumnarExplainer<T extends Output<T>> {

    /**
     * Explains the supplied data. The Map is first converted into
     * an {@link Example} using a {@link RowProcessor}, before being
     * supplied to the internal {@link Model}.
     * @param input The data to explain.
     * @return An Explanation for this data.
     */
    public Explanation<T> explain(Map<String,String> input);

}
