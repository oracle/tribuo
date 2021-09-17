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

package org.tribuo.provenance;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;

/**
 * A tag interface for trainer provenances.
 */
public interface TrainerProvenance extends ConfiguredObjectProvenance {

    /**
     * The name of the provenance field recording the train invocation count.
     */
    public static final String TRAIN_INVOCATION_COUNT = "train-invocation-count";

    /**
     * The name of the provenance field recording if this is a sequence trainer.
     */
    public static final String IS_SEQUENCE = DatasetProvenance.IS_SEQUENCE;

    /**
     * The name of the provenance field recording the Tribuo version used in training.
     */
    public static final String TRIBUO_VERSION_STRING = "tribuo-version";

}
