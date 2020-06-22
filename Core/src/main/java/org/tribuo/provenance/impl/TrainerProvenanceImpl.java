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

package org.tribuo.provenance.impl;

import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.provenance.SkeletalTrainerProvenance;
import org.tribuo.provenance.TrainerProvenance;
import org.tribuo.sequence.SequenceTrainer;

import java.util.Map;

/**
 * An implementation of {@link TrainerProvenance} that delegates everything to
 * {@link SkeletalTrainerProvenance}. Used for trainers which don't
 * require additional information stored beyond their configurable parameters
 * and the standard trainer instance parameters.
 */
public final class TrainerProvenanceImpl extends SkeletalTrainerProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * Construct a TrainerProvenance by reading all the configurable parameters
     * along with the train call count.
     * @param host The trainer to inspect.
     * @param <T> The type of the {@link Output}.
     */
    public <T extends Output<T>> TrainerProvenanceImpl(Trainer<T> host) {
        super(host);
    }

    /**
     * Construct a TrainerProvenance by reading all the configurable parameters
     * along with the train call count.
     * @param host The sequence trainer to inspect.
     * @param <T> The type of the {@link Output}.
     */
    public <T extends Output<T>> TrainerProvenanceImpl(SequenceTrainer<T> host) {
        super(host);
    }

    /**
     * Construct a TrainerProvenance by extracting the necessary fields from the supplied
     * map.
     * @param map The serialised form of this provenance.
     */
    public TrainerProvenanceImpl(Map<String, Provenance> map) {
        super(extractProvenanceInfo(map));
    }
}
