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
 * Provides classes and infrastructure for anomaly detection problems.
 * <p>
 * The anomaly detection {@link org.tribuo.Output} implementation is called {@link org.tribuo.anomaly.Event}
 * and has associated Evaluators, OutputFactory and OutputInfos.
 * </p>
 * <p>
 * Event trainers are allowed to throw IllegalArgumentException if they are supplied
 * an {@link org.tribuo.anomaly.Event.EventType#ANOMALOUS} at training time. It's noted in the documentation if they
 * do support training from anomalous and expected data.
 * </p>
 * <p>
 * Note: This implementation is effectively a specialised binary classification problem.
 * </p>
 */
package org.tribuo.anomaly;