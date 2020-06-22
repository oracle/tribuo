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

/**
 * An explainer for text data. Hopefully uses a sensible sampling mechanism that understands text.
 */
public interface TextExplainer<T extends Output<T>> {

    /**
     * Converts the supplied text into an {@link Example}, and
     * generates an explanation of the contained {@link Model}'s prediction.
     * @param inputText The text to explain.
     * @return An explanation of the prediction on this input text.
     */
    public Explanation<T> explain(String inputText);

}
