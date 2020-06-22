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

import com.oracle.labs.mlrg.olcut.provenance.MapProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Tribuo;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

/**
 * Contains provenance information for an instance of a {@link org.tribuo.Model}.
 * <p>
 * Made up of the class name of the model object, the date and time it was trained, the provenance of
 * the training data, and the provenance of the trainer.
 */
public class ModelProvenance implements ObjectProvenance {
    private static final long serialVersionUID = 1L;

    protected static final String DATASET = "dataset";
    protected static final String TRAINER = "trainer";
    protected static final String TRAINING_TIME = "trained-at";
    protected static final String INSTANCE_VALUES = "instance-values";
    private static final String TRIBUO_VERSION_STRING = "tribuo-version";

    protected final String className;

    protected final OffsetDateTime time;

    protected final DatasetProvenance datasetProvenance;

    protected final TrainerProvenance trainerProvenance;

    protected final MapProvenance<? extends Provenance> instanceProvenance;

    protected final String versionString;

    public ModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance, TrainerProvenance trainerProvenance) {
        this.className = className;
        this.time = time;
        this.datasetProvenance = datasetProvenance;
        this.trainerProvenance = trainerProvenance;
        this.instanceProvenance = new MapProvenance<>();
        this.versionString = Tribuo.VERSION;
    }

    public ModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance, TrainerProvenance trainerProvenance, Map<String,Provenance> instanceProvenance) {
        this.className = className;
        this.time = time;
        this.datasetProvenance = datasetProvenance;
        this.trainerProvenance = trainerProvenance;
        this.instanceProvenance = new MapProvenance<>(instanceProvenance);
        this.versionString = Tribuo.VERSION;
    }

    public ModelProvenance(Map<String,Provenance> map) {
        this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class, ModelProvenance.class.getSimpleName()).getValue();
        this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET,DatasetProvenance.class, ModelProvenance.class.getSimpleName());
        this.trainerProvenance = ObjectProvenance.checkAndExtractProvenance(map,TRAINER,TrainerProvenance.class, ModelProvenance.class.getSimpleName());
        this.time = ObjectProvenance.checkAndExtractProvenance(map,TRAINING_TIME,DateTimeProvenance.class, ModelProvenance.class.getSimpleName()).getValue();
        this.instanceProvenance = (MapProvenance<?>) ObjectProvenance.checkAndExtractProvenance(map,INSTANCE_VALUES,MapProvenance.class, ModelProvenance.class.getSimpleName());
        this.versionString = ObjectProvenance.checkAndExtractProvenance(map, TRIBUO_VERSION_STRING,StringProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
    }

    /**
     * The training timestamp.
     * @return The timestamp.
     */
    public OffsetDateTime getTrainingTime() {
        return time;
    }

    /**
     * The training dataset provenance.
     * @return The training dataset provenance.
     */
    public DatasetProvenance getDatasetProvenance() {
        return datasetProvenance;
    }

    /**
     * The trainer provenance.
     * @return The trainer provenance.
     */
    public TrainerProvenance getTrainerProvenance() {
        return trainerProvenance;
    }

    /**
     * Provenance for the specific training run which created this model.
     * @return The instance provenance.
     */
    public MapProvenance<? extends Provenance> getInstanceProvenance() {
        return instanceProvenance;
    }

    /**
     * The Tribuo version used to create this dataset.
     * @return The Tribuo version.
     */
    public String getTribuoVersion() {
        return versionString;
    }

    @Override
    public String toString() {
        return generateString("Model");
    }

    @Override
    public String getClassName() {
        return className;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ModelProvenance)) return false;
        ModelProvenance pairs = (ModelProvenance) o;
        return className.equals(pairs.className) &&
                time.equals(pairs.time) &&
                datasetProvenance.equals(pairs.datasetProvenance) &&
                trainerProvenance.equals(pairs.trainerProvenance) &&
                instanceProvenance.equals(pairs.instanceProvenance) &&
                versionString.equals(pairs.versionString);
    }

    @Override
    public int hashCode() {
        return Objects.hash(className, time, datasetProvenance, trainerProvenance, instanceProvenance, versionString);
    }

    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        ArrayList<Pair<String,Provenance>> iterable = new ArrayList<>();
        iterable.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,className)));
        iterable.add(new Pair<>(DATASET,datasetProvenance));
        iterable.add(new Pair<>(TRAINER,trainerProvenance));
        iterable.add(new Pair<>(TRAINING_TIME,new DateTimeProvenance(TRAINING_TIME,time)));
        iterable.add(new Pair<>(INSTANCE_VALUES,instanceProvenance));
        iterable.add(new Pair<>(TRIBUO_VERSION_STRING,new StringProvenance(TRIBUO_VERSION_STRING,versionString)));
        return iterable.iterator();
    }
}
