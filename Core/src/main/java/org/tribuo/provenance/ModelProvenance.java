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
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Contains provenance information for an instance of a {@link org.tribuo.Model}.
 * <p>
 * Made up of the class name of the model object, the date and time it was trained, the provenance of
 * the training data, and the provenance of the trainer.
 * <p>
 * In addition by default it collects the Java version, OS name and system architecture, along
 * with the running Tribuo version.
 */
public class ModelProvenance implements ObjectProvenance {
    private static final long serialVersionUID = 1L;

    protected static final String DATASET = "dataset";
    protected static final String TRAINER = "trainer";
    protected static final String TRAINING_TIME = "trained-at";
    protected static final String INSTANCE_VALUES = "instance-values";
    protected static final String TRIBUO_VERSION_STRING = "tribuo-version";

    // Note these have been added due to a discrepancy between java.lang.Math
    // and java.lang.StrictMath on x64 and aarch64 platforms (and between Java 8 and 9+).
    // Training a linear SGD predictor can create different models on different platforms
    // due to this discrepancy.
    protected static final String JAVA_VERSION_STRING = "java-version";
    protected static final String OS_STRING = "os-name";
    protected static final String ARCH_STRING = "os-arch";

    protected static final String UNKNOWN_VERSION = "unknown-version";

    protected final String className;

    protected final OffsetDateTime time;

    protected final DatasetProvenance datasetProvenance;

    protected final TrainerProvenance trainerProvenance;

    protected final MapProvenance<? extends Provenance> instanceProvenance;

    protected final String versionString;

    protected final String javaVersionString;

    protected final String osString;

    protected final String archString;

    /**
     * Creates a model provenance tracking the class name, creation time, dataset provenance and trainer provenance.
     * <p>
     * Also tracks system details like the os name, os architecture, java version, and Tribuo version.
     * @param className The model class name.
     * @param time The model creation time.
     * @param datasetProvenance The dataset provenance.
     * @param trainerProvenance The trainer provenance.
     */
    public ModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance,
                           TrainerProvenance trainerProvenance) {
        this(className,time,datasetProvenance,trainerProvenance,Collections.emptyMap());
    }

    /**
     * Creates a model provenance tracking the class name, creation time, dataset provenance,
     * trainer provenance and any instance specific provenance.
     * <p>
     * Also tracks system details like the os name, os architecture, java version, and Tribuo version.
     * @param className The model class name.
     * @param time The model creation time.
     * @param datasetProvenance The dataset provenance.
     * @param trainerProvenance The trainer provenance.
     * @param instanceProvenance Provenance for this specific model training run.
     */
    public ModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance,
                           TrainerProvenance trainerProvenance, Map<String,Provenance> instanceProvenance) {
        this(className,time,datasetProvenance,trainerProvenance,instanceProvenance,true);
    }

    /**
     * Creates a model provenance tracking the class name, creation time, dataset provenance,
     * trainer provenance and any instance specific provenance.
     * <p>
     * Also optionally tracks system details like the os name, os architecture, java version, and Tribuo version.
     * @param className The model class name.
     * @param time The model creation time.
     * @param datasetProvenance The dataset provenance.
     * @param trainerProvenance The trainer provenance.
     * @param instanceProvenance Provenance for this specific model training run.
     * @param trackSystem If true then store the java version, os name and os arch in the provenance.
     */
    public ModelProvenance(String className, OffsetDateTime time, DatasetProvenance datasetProvenance,
                           TrainerProvenance trainerProvenance, Map<String,Provenance> instanceProvenance,
                           boolean trackSystem) {
        this.className = className;
        this.time = time;
        this.datasetProvenance = datasetProvenance;
        this.trainerProvenance = trainerProvenance;
        this.instanceProvenance = instanceProvenance.isEmpty() ? new MapProvenance<>() : new MapProvenance<>(instanceProvenance);
        this.versionString = Tribuo.VERSION;
        if (trackSystem) {
            this.javaVersionString = System.getProperty("java.version");
            this.osString = System.getProperty("os.name");
            this.archString = System.getProperty("os.arch");
        } else {
            this.javaVersionString = UNKNOWN_VERSION;
            this.osString = UNKNOWN_VERSION;
            this.archString = UNKNOWN_VERSION;
        }
    }

    /**
     * Used by the provenance unmarshalling system.
     * <p>
     * Throws {@link com.oracle.labs.mlrg.olcut.provenance.ProvenanceException} if there are missing
     * fields.
     * @param map The provenance map.
     */
    public ModelProvenance(Map<String,Provenance> map) {
        this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class, ModelProvenance.class.getSimpleName()).getValue();
        this.datasetProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASET,DatasetProvenance.class, ModelProvenance.class.getSimpleName());
        this.trainerProvenance = ObjectProvenance.checkAndExtractProvenance(map,TRAINER,TrainerProvenance.class, ModelProvenance.class.getSimpleName());
        this.time = ObjectProvenance.checkAndExtractProvenance(map,TRAINING_TIME,DateTimeProvenance.class, ModelProvenance.class.getSimpleName()).getValue();
        this.instanceProvenance = (MapProvenance<?>) ObjectProvenance.checkAndExtractProvenance(map,INSTANCE_VALUES,MapProvenance.class, ModelProvenance.class.getSimpleName());
        this.versionString = ObjectProvenance.checkAndExtractProvenance(map,TRIBUO_VERSION_STRING,StringProvenance.class,ModelProvenance.class.getSimpleName()).getValue();

        // Provenances added in Tribuo 4.1
        this.javaVersionString = ObjectProvenance.maybeExtractProvenance(map,JAVA_VERSION_STRING,StringProvenance.class,ModelProvenance.class.getSimpleName()).map(StringProvenance::getValue).orElse(UNKNOWN_VERSION);
        this.osString = ObjectProvenance.maybeExtractProvenance(map,OS_STRING,StringProvenance.class,ModelProvenance.class.getSimpleName()).map(StringProvenance::getValue).orElse(UNKNOWN_VERSION);
        this.archString = ObjectProvenance.maybeExtractProvenance(map,ARCH_STRING,StringProvenance.class,ModelProvenance.class.getSimpleName()).map(StringProvenance::getValue).orElse(UNKNOWN_VERSION);
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

    /**
     * The Java version used to create this model.
     * @return The Java version.
     */
    public String getJavaVersion() {
        return javaVersionString;
    }

    /**
     * The name of the OS used to create this model.
     * @return The OS name.
     */
    public String getOS() {
        return osString;
    }

    /**
     * The CPU architecture used to create this model.
     * @return The CPU architecture.
     */
    public String getArch() {
        return archString;
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
                versionString.equals(pairs.versionString) &&
                javaVersionString.equals(pairs.javaVersionString) &&
                osString.equals(pairs.osString) &&
                archString.equals(pairs.archString);
    }

    @Override
    public int hashCode() {
        return Objects.hash(className, time, datasetProvenance, trainerProvenance, instanceProvenance, versionString);
    }

    /**
     * Returns a list of all the provenances in this model provenance so subclasses
     * can append to the list.
     * @return A list of all the provenances in this class.
     */
    protected List<Pair<String,Provenance>> internalProvenances() {
        ArrayList<Pair<String,Provenance>> iterable = new ArrayList<>();
        iterable.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,className)));
        iterable.add(new Pair<>(DATASET,datasetProvenance));
        iterable.add(new Pair<>(TRAINER,trainerProvenance));
        iterable.add(new Pair<>(TRAINING_TIME,new DateTimeProvenance(TRAINING_TIME,time)));
        iterable.add(new Pair<>(INSTANCE_VALUES,instanceProvenance));
        iterable.add(new Pair<>(TRIBUO_VERSION_STRING,new StringProvenance(TRIBUO_VERSION_STRING,versionString)));
        iterable.add(new Pair<>(JAVA_VERSION_STRING,new StringProvenance(JAVA_VERSION_STRING,javaVersionString)));
        iterable.add(new Pair<>(OS_STRING,new StringProvenance(OS_STRING,osString)));
        iterable.add(new Pair<>(ARCH_STRING,new StringProvenance(ARCH_STRING,archString)));
        return iterable;
    }

    /**
     * Calls {@link #internalProvenances()} and returns the iterator from that list.
     * @return An iterator over all the provenances.
     */
    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        return internalProvenances().iterator();
    }
}
