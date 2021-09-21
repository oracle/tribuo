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
import org.tribuo.Trainer;
import org.tribuo.provenance.TrainerProvenance;

import java.util.Collections;
import java.util.Map;

/**
 * An empty TrainerProvenance, should not be used except by the provenance removal system.
 */
public final class EmptyTrainerProvenance implements TrainerProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * Constructs an empty trainer provenance.
     */
    public EmptyTrainerProvenance() { }

    /**
     * Deserialization constructor.
     * @param map The provenances, which are ignored as this provenance is empty.
     */
    public EmptyTrainerProvenance(Map<String,Provenance> map) {}

    @Override
    public Map<String, Provenance> getConfiguredParameters() {
        return Collections.emptyMap();
    }

    @Override
    public String getClassName() {
        return Trainer.class.getName();
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof EmptyTrainerProvenance;
    }

    @Override
    public int hashCode() {
        return 31;
    }

    @Override
    public String toString() {
        return generateString("EmptyTrainer");
    }
}
