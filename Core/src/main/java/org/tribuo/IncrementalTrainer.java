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

package org.tribuo;

/**
 * An interface for incremental training of {@link Model}s.
 */
public interface IncrementalTrainer<T extends Output<T>, U extends Model<T>> extends Trainer<T> {

    /**
     * Incrementally trains the supplied model with the new data.
     * @param newData The additional training data.
     * @param model The model to update.
     * @return The updated model.
     */
    public U incrementalTrain(Dataset<T> newData, U model);

}
