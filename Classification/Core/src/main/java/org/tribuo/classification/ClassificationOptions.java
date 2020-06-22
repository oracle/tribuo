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

package org.tribuo.classification;

import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Trainer;

/**
 * An {@link Options} that can produce a classification {@link Trainer} based on the
 * provided arguments.
 * @param <TRAINER> The type of the trainer produced.
 */
public interface ClassificationOptions<TRAINER extends Trainer<Label>> extends Options {

    /**
     * Constructs the trainer based on the provided arguments.
     * @return The trainer.
     */
    TRAINER getTrainer();

}
