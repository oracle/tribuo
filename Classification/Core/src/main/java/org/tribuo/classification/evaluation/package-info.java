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
 * Evaluation classes for multi-class classification. Also used as the basis for
 * the multi-label classification evaluation package.
 * <p>
 * The default metrics calculated are found in {@link org.tribuo.classification.evaluation.LabelMetrics},
 * and are based on statistics calculated from a {@link org.tribuo.classification.evaluation.ConfusionMatrix}.
 * <p>
 * User specified metrics are not currently supported.
 */
package org.tribuo.classification.evaluation;