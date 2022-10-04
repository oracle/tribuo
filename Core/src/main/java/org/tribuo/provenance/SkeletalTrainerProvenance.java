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

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceException;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.Output;
import org.tribuo.Trainer;
import org.tribuo.Tribuo;
import org.tribuo.sequence.SequenceTrainer;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The skeleton of a TrainerProvenance that extracts the configured parameters.
 */
public abstract class SkeletalTrainerProvenance extends SkeletalConfiguredObjectProvenance implements TrainerProvenance {
    private static final long serialVersionUID = 1L;

    private final IntProvenance invocationCount;

    private final BooleanProvenance isSequence;

    private final StringProvenance version;

    /**
     * Builds a trainer provenance extracting the standard information from the host.
     * @param host The host object.
     * @param <T> The output type.
     */
    protected <T extends Output<T>> SkeletalTrainerProvenance(Trainer<T> host) {
        super(host,"Trainer");
        this.isSequence = new BooleanProvenance(IS_SEQUENCE,false);
        this.invocationCount = new IntProvenance(TRAIN_INVOCATION_COUNT,host.getInvocationCount());
        this.version = new StringProvenance(TRIBUO_VERSION_STRING, Tribuo.VERSION);
    }

    /**
     * Builds a trainer provenance extracting the standard information from the host.
     * @param host The host object.
     * @param <T> The output type.
     */
    protected <T extends Output<T>> SkeletalTrainerProvenance(SequenceTrainer<T> host) {
        super(host,"SequenceTrainer");
        this.isSequence = new BooleanProvenance(IS_SEQUENCE,true);
        this.invocationCount = new IntProvenance(TRAIN_INVOCATION_COUNT,host.getInvocationCount());
        this.version = new StringProvenance(TRIBUO_VERSION_STRING, Tribuo.VERSION);
    }

    /**
     * Deserialization constructor.
     * @param map The provenance map.
     */
    protected SkeletalTrainerProvenance(Map<String, Provenance> map) {
        this(extractProvenanceInfo(map));
    }

    /**
     * Deserialization constructor.
     * @param info The extracted provenance information.
     */
    protected SkeletalTrainerProvenance(ExtractedInfo info) {
        super(info);
        this.invocationCount = SkeletalConfiguredObjectProvenance.checkAndExtractProvenance(info,TRAIN_INVOCATION_COUNT,IntProvenance.class, info.className);
        this.isSequence = SkeletalConfiguredObjectProvenance.checkAndExtractProvenance(info,IS_SEQUENCE,BooleanProvenance.class, info.className);
        this.version = SkeletalConfiguredObjectProvenance.checkAndExtractProvenance(info,TRIBUO_VERSION_STRING,StringProvenance.class, info.className);
    }

    /**
     * Is this a sequence trainer.
     * @return True if it's a sequence trainer.
     */
    public boolean isSequence() {
        return isSequence.getValue();
    }

    /**
     * The Tribuo version.
     * @return The Tribuo version.
     */
    public String getTribuoVersion() {
        return version.getValue();
    }

    @Override
    public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
        Map<String, PrimitiveProvenance<?>> map = super.getInstanceValues();

        map.put(TRAIN_INVOCATION_COUNT, invocationCount);
        map.put(IS_SEQUENCE, isSequence);
        map.put(TRIBUO_VERSION_STRING, version);

        return map;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SkeletalTrainerProvenance)) return false;
        if (!super.equals(o)) return false;
        SkeletalTrainerProvenance pairs = (SkeletalTrainerProvenance) o;
        return invocationCount.equals(pairs.invocationCount) &&
                isSequence.equals(pairs.isSequence);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), invocationCount, isSequence);
    }

    /**
     * Extracts the provenance information from the supplied map, splitting it into configuration and instance information.
     * @param map The provenance map.
     * @return The provenance information.
     */
    protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
        String className;
        String hostTypeStringName;
        Map<String,Provenance> configuredParameters = new HashMap<>(map);
        Map<String,PrimitiveProvenance<?>> instanceValues = new HashMap<>();
        if (configuredParameters.containsKey(ObjectProvenance.CLASS_NAME)) {
            className = configuredParameters.remove(ObjectProvenance.CLASS_NAME).toString();
        } else {
            throw new ProvenanceException("Failed to find class name when constructing SkeletalTrainerProvenance");
        }
        if (configuredParameters.containsKey(SkeletalConfiguredObjectProvenance.HOST_SHORT_NAME)) {
            hostTypeStringName = configuredParameters.remove(SkeletalConfiguredObjectProvenance.HOST_SHORT_NAME).toString();
        } else {
            throw new ProvenanceException("Failed to find host type short name when constructing SkeletalTrainerProvenance");
        }
        if (configuredParameters.containsKey(TrainerProvenance.TRAIN_INVOCATION_COUNT)) {
            Provenance tmpProv = configuredParameters.remove(TrainerProvenance.TRAIN_INVOCATION_COUNT);
            if (tmpProv instanceof IntProvenance) {
                instanceValues.put(TRAIN_INVOCATION_COUNT,(IntProvenance) tmpProv);
            } else {
                throw new ProvenanceException(TRAIN_INVOCATION_COUNT + " was not of type IntProvenance in class " + className);
            }
        } else {
            throw new ProvenanceException("Failed to find invocation count when constructing SkeletalTrainerProvenance");
        }
        if (configuredParameters.containsKey(TrainerProvenance.IS_SEQUENCE)) {
            Provenance tmpProv = configuredParameters.remove(TrainerProvenance.IS_SEQUENCE);
            if (tmpProv instanceof BooleanProvenance) {
                instanceValues.put(IS_SEQUENCE,(BooleanProvenance) tmpProv);
            } else {
                throw new ProvenanceException(IS_SEQUENCE + " was not of type BooleanProvenance in class " + className);
            }
        } else {
            throw new ProvenanceException("Failed to find is-sequence when constructing SkeletalTrainerProvenance");
        }
        if (configuredParameters.containsKey(TrainerProvenance.TRIBUO_VERSION_STRING)) {
            Provenance tmpProv = configuredParameters.remove(TrainerProvenance.TRIBUO_VERSION_STRING);
            if (tmpProv instanceof StringProvenance) {
                instanceValues.put(TRIBUO_VERSION_STRING,(StringProvenance) tmpProv);
            } else {
                throw new ProvenanceException(TRIBUO_VERSION_STRING + " was not of type StringProvenance in class " + className);
            }
        } else {
            throw new ProvenanceException("Failed to find Tribuo version when constructing SkeletalTrainerProvenance");
        }

        return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceValues);
    }
}
