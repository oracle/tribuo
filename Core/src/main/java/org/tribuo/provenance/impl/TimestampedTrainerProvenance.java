/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceException;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.Trainer;
import org.tribuo.Tribuo;
import org.tribuo.provenance.TrainerProvenance;

import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A TrainerProvenance with a timestamp, used when there was no trainer
 * involved in model construction (e.g., creating an {@link org.tribuo.ensemble.EnsembleModel}
 * from existing models).
 */
public final class TimestampedTrainerProvenance implements TrainerProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * The name of the provenance field storing the model creation time.
     */
    public static final String CREATION_TIME = "creation-time";

    private final DateTimeProvenance creationTime;
    private final StringProvenance version;

    /**
     * Creates a TimestampedTrainerProvenance, tracking the creation time and Tribuo version.
     */
    public TimestampedTrainerProvenance() {
        this.creationTime = new DateTimeProvenance(CREATION_TIME, OffsetDateTime.now());
        this.version = new StringProvenance(TRIBUO_VERSION_STRING, Tribuo.VERSION);
    }

    /**
     * Used for deserializing provenances from the marshalled form.
     * @param map The provenance map.
     */
    public TimestampedTrainerProvenance(Map<String,Provenance> map) {
        Provenance tmp = map.get(TRIBUO_VERSION_STRING);
        if (tmp != null) {
            if (StringProvenance.class.isInstance(tmp)) {
                this.version = (StringProvenance) tmp;
            } else {
                throw new ProvenanceException("Failed to cast " + TRIBUO_VERSION_STRING + " when constructing TimestampedTrainerProvenance, found " + tmp);
            }
        } else {
            throw new ProvenanceException("Failed to find " + TRIBUO_VERSION_STRING + " when constructing TimestampedTrainerProvenance");
        }
        tmp = map.get(CREATION_TIME);
        if (tmp != null) {
            if (DateTimeProvenance.class.isInstance(tmp)) {
                this.creationTime = (DateTimeProvenance) tmp;
            } else {
                throw new ProvenanceException("Failed to cast " + CREATION_TIME + " when constructing TimestampedTrainerProvenance, found " + tmp);
            }
        } else {
            throw new ProvenanceException("Failed to find " + CREATION_TIME + " when constructing TimestampedTrainerProvenance");
        }
    }

    @Override
    public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
        Map<String,PrimitiveProvenance<?>> provMap = new HashMap<>();
        provMap.put(CREATION_TIME,creationTime);
        provMap.put(TRIBUO_VERSION_STRING,version);
        return provMap;
    }

    @Override
    public Map<String, Provenance> getConfiguredParameters() {
        return Collections.emptyMap();
    }

    @Override
    public String getClassName() {
        return Trainer.class.getName();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TimestampedTrainerProvenance pairs = (TimestampedTrainerProvenance) o;
        return creationTime.equals(pairs.creationTime) && version.equals(pairs.version);
    }

    @Override
    public int hashCode() {
        return Objects.hash(creationTime, version);
    }

    @Override
    public String toString() {
        return generateString("Trainer");
    }
}
