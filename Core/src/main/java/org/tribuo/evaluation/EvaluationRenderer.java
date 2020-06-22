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

package org.tribuo.evaluation;

import org.tribuo.Output;

/**
 * Renders an {@link Evaluation} into a String.
 * <p>
 * For example, an implementation might produce
 *
 * @param <T> type of output
 * @param <E> type of evaluation
 */
@FunctionalInterface
public interface EvaluationRenderer<T extends Output<T>, E extends Evaluation<T>> {

    /**
     * Convert the evaluation to a string. For example:
     *
     * <pre>
     *     EvaluationRenderer&lt;Label, LabelEvaluation&gt; renderer = (LabelEvaluation eval) -&gt; {
     *         StringBuilder sb = new StringBuilder();
     *         sb.append(String.format("macro F1: %.2f", eval.macroAveragedF1()));
     *         sb.append(String.format("micro F1: %.2f", eval.microAveragedF1()));
     *         return sb.toString();
     *     }
     *
     *     LabelEvaluation evaluation = ...
     *     System.out.println(renderer.apply(evaluation));
     * </pre>
     *
     * @param evaluation The evaluation to render.
     * @return The renderer's representation of the evaluation as a String.
     */
    String apply(E evaluation);

}