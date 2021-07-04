/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.evaluation;

import org.tribuo.classification.evaluation.ClassifierEvaluation;
import org.tribuo.multilabel.MultiLabel;

/**
 * A {@link MultiLabel} specific {@link ClassifierEvaluation}.
 * <p>
 * Used to hold multi-label specific evaluation metrics.
 */
public interface MultiLabelEvaluation extends ClassifierEvaluation<MultiLabel> {

    /**
     * The average across the predictions of the intersection of the true and predicted labels divided by the
     * union of the true and predicted labels.
     * @return The Jaccard score.
     */
    public double jaccardScore();

}
