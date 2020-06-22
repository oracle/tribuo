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
import org.tribuo.Output;
import org.tribuo.classification.Label;

/**
 * An explainer for tabular data.
 */
public interface TabularExplainer<T extends Output<T>> {

    /**
     * Explain why the supplied {@link Example} is classified a certain way.
     * @param example The Example to explain.
     * @return An Explanation for this example.
     */
    public Explanation<T> explain(Example<Label> example);

}
