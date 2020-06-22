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
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

/**
 * Model provenance for ensemble models.
 */
public class EnsembleModelProvenance extends ModelProvenance {
    private static final long serialVersionUID = 1L;

    protected static final String MEMBERS = "member-provenance";

    private final ListProvenance<? extends ModelProvenance> memberProvenance;

    public EnsembleModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance, TrainerProvenance trainerProvenance, ListProvenance<? extends ModelProvenance> memberProvenance) {
        super(className, time, datasetProvenance, trainerProvenance);
        this.memberProvenance = memberProvenance;
    }

    public EnsembleModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance, TrainerProvenance trainerProvenance, Map<String, Provenance> instanceProvenance, ListProvenance<? extends ModelProvenance> memberProvenance) {
        super(className, time, datasetProvenance, trainerProvenance, instanceProvenance);
        this.memberProvenance = memberProvenance;
    }

    @SuppressWarnings("unchecked") // member provenance cast.
    public EnsembleModelProvenance(Map<String, Provenance> map) {
        super(map);
        this.memberProvenance = (ListProvenance<? extends ModelProvenance>) ObjectProvenance.checkAndExtractProvenance(map,MEMBERS,ListProvenance.class, EnsembleModelProvenance.class.getSimpleName());
    }

    public ListProvenance<? extends ModelProvenance> getMemberProvenance() {
        return memberProvenance;
    }

    @Override
    public String toString() {
        return generateString("EnsembleModel");
    }

    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        ArrayList<Pair<String,Provenance>> iterable = new ArrayList<>();
        iterable.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,className)));
        iterable.add(new Pair<>(DATASET,datasetProvenance));
        iterable.add(new Pair<>(TRAINER,trainerProvenance));
        iterable.add(new Pair<>(TRAINING_TIME,new DateTimeProvenance(TRAINING_TIME,time)));
        iterable.add(new Pair<>(INSTANCE_VALUES,instanceProvenance));
        iterable.add(new Pair<>(MEMBERS,memberProvenance));
        return iterable.iterator();
    }
}
