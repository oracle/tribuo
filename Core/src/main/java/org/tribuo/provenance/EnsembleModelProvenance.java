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

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.util.Pair;

import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Model provenance for ensemble models.
 */
public class EnsembleModelProvenance extends ModelProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * The name of the provenance field where the member provenances are stored.
     */
    protected static final String MEMBERS = "member-provenance";

    private final ListProvenance<? extends ModelProvenance> memberProvenance;

    /**
     * Creates a provenance for an ensemble model tracking the class name, creation time, dataset provenance and
     * trainer provenance along with the individual model provenances
     * for each ensemble member.
     * <p>
     * Also tracks system details like the os name, os architecture, java version, and Tribuo version.
     * @param className The model class name.
     * @param time The model creation time.
     * @param datasetProvenance The dataset provenance.
     * @param trainerProvenance The trainer provenance.
     * @param memberProvenance The ensemble member provenances.
     */
    public EnsembleModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance,
                                   TrainerProvenance trainerProvenance,
                                   ListProvenance<? extends ModelProvenance> memberProvenance) {
        super(className, time, datasetProvenance, trainerProvenance);
        this.memberProvenance = memberProvenance;
    }

    /**
     * Creates a provenance for an ensemble model tracking the class name, creation time, dataset provenance,
     * trainer provenance and any instance specific provenance along with the individual model provenances
     * for each ensemble member.
     * <p>
     * Also tracks system details like the os name, os architecture, java version, and Tribuo version.
     * @param className The model class name.
     * @param time The model creation time.
     * @param datasetProvenance The dataset provenance.
     * @param trainerProvenance The trainer provenance.
     * @param instanceProvenance Provenance for this specific model training run.
     * @param memberProvenance The ensemble member provenances.
     */
    public EnsembleModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance,
                                   TrainerProvenance trainerProvenance, Map<String, Provenance> instanceProvenance,
                                   ListProvenance<? extends ModelProvenance> memberProvenance) {
        this(className,time,datasetProvenance,trainerProvenance,instanceProvenance,true,memberProvenance);
    }

    /**
     * Creates a provenance for an ensemble model tracking the class name, creation time, dataset provenance,
     * trainer provenance and any instance specific provenance along with the individual model provenances
     * for each ensemble member.
     * <p>
     * Also optionally tracks system details like the os name, os architecture, java version, and Tribuo version.
     * @param className The model class name.
     * @param time The model creation time.
     * @param datasetProvenance The dataset provenance.
     * @param trainerProvenance The trainer provenance.
     * @param instanceProvenance Provenance for this specific model training run.
     * @param memberProvenance The ensemble member provenances.
     * @param trackSystem If true then store the java version, os name and os arch in the provenance.
     */
    public EnsembleModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance,
                                   TrainerProvenance trainerProvenance, Map<String, Provenance> instanceProvenance,
                                   boolean trackSystem, ListProvenance<? extends ModelProvenance> memberProvenance) {
        super(className, time, datasetProvenance, trainerProvenance, instanceProvenance, trackSystem);
        this.memberProvenance = memberProvenance;
    }

    /**
     * Used by the provenance unmarshalling system.
     * <p>
     * Throws {@link com.oracle.labs.mlrg.olcut.provenance.ProvenanceException} if there are missing
     * fields.
     * @param map The provenance map.
     */
    @SuppressWarnings("unchecked") // member provenance cast.
    public EnsembleModelProvenance(Map<String, Provenance> map) {
        super(map);
        this.memberProvenance = (ListProvenance<? extends ModelProvenance>) ObjectProvenance.checkAndExtractProvenance(map,MEMBERS,ListProvenance.class, EnsembleModelProvenance.class.getSimpleName());
    }

    /**
     * Get the provenances for each ensemble member.
     * @return The member provenances.
     */
    public ListProvenance<? extends ModelProvenance> getMemberProvenance() {
        return memberProvenance;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        EnsembleModelProvenance pairs = (EnsembleModelProvenance) o;
        return memberProvenance.equals(pairs.memberProvenance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode(), memberProvenance);
    }

    @Override
    public String toString() {
        return generateString("EnsembleModel");
    }

    @Override
    protected List<Pair<String, Provenance>> internalProvenances() {
        List<Pair<String, Provenance>> superList = super.internalProvenances();
        superList.add(new Pair<>(MEMBERS,memberProvenance));
        return superList;
    }
}
