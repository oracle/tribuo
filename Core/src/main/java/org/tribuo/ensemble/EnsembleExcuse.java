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

package org.tribuo.ensemble;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Output;
import org.tribuo.Prediction;

import java.util.List;
import java.util.Map;

/**
 * An {@link Excuse} which has a List of excuses for each of the ensemble members.
 */
public class EnsembleExcuse<T extends Output<T>> extends Excuse<T> {

    private final List<Excuse<T>> innerExcuses;

    /**
     * Constructs an ensemble excuse, comprising the excuses from each ensemble member, along with the feature weights.
     * @param example The example.
     * @param prediction The prediction to excuse.
     * @param weights The weights.
     * @param innerExcuses The ensemble member excuses.
     */
    public EnsembleExcuse(Example<T> example, Prediction<T> prediction, Map<String,List<Pair<String,Double>>> weights, List<Excuse<T>> innerExcuses) {
        super(example,prediction,weights);
        this.innerExcuses = innerExcuses;
    }

    /**
     * The individual ensemble member's excuses.
     * @return The individual excuses.
     */
    public List<Excuse<T>> getInnerExcuses() {
        return innerExcuses;
    }

}
