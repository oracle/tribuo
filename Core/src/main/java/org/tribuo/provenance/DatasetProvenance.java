/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Dataset;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.Tribuo;
import org.tribuo.sequence.MutableSequenceDataset;
import org.tribuo.sequence.SequenceDataset;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Base class for dataset provenance.
 * <p>
 * Dataset provenance can be a chain of other DataProvenances which track operations like selection
 * and subsampling.
 * </p>
 */
public class DatasetProvenance implements DataProvenance, ObjectProvenance {
    private static final long serialVersionUID = 1L;

    static final String DATASOURCE = "datasource";
    static final String TRANSFORMATIONS = "transformations";
    static final String IS_DENSE = "is-dense";
    static final String IS_SEQUENCE = "is-sequence";
    static final String NUM_EXAMPLES = "num-examples";
    static final String NUM_FEATURES = "num-features";
    static final String NUM_OUTPUTS = "num-outputs";
    private static final String TRIBUO_VERSION_STRING = "tribuo-version";

    private final String className;

    private final DataProvenance sourceProvenance;

    private final ListProvenance<ObjectProvenance> transformationProvenance;

    private final boolean isDense;

    private final boolean isSequence;

    private final int numExamples;

    private final int numFeatures;

    private final int numOutputs;

    private final String versionString;

    /**
     * Creates a dataset provenance from the supplied dataset.
     * @param sourceProvenance The data source provenance.
     * @param transformationProvenance The transformations applied to this dataset.
     * @param dataset The dataset itself.
     * @param <T> The output type of the dataset.
     */
    public <T extends Output<T>> DatasetProvenance(DataProvenance sourceProvenance, ListProvenance<ObjectProvenance> transformationProvenance, Dataset<T> dataset) {
        this(sourceProvenance,transformationProvenance,dataset.getClass().getName(),dataset instanceof MutableDataset && ((MutableDataset<T>) dataset).isDense(),false,dataset.size(),dataset.getFeatureMap().size(),dataset.getOutputInfo().size());
    }

    /**
     * Creates a dataset provenance from the supplied sequence dataset.
     * @param sourceProvenance The data source provenance.
     * @param transformationProvenance The transformations applied to this sequence dataset.
     * @param dataset The sequence dataset itself.
     * @param <T> The output type of the sequence dataset.
     */
    public <T extends Output<T>> DatasetProvenance(DataProvenance sourceProvenance, ListProvenance<ObjectProvenance> transformationProvenance, SequenceDataset<T> dataset) {
        this(sourceProvenance,transformationProvenance,dataset.getClass().getName(),dataset instanceof MutableSequenceDataset && ((MutableSequenceDataset<T>) dataset).isDense(),true,dataset.size(),dataset.getFeatureMap().size(),dataset.getOutputInfo().size());
    }

    /**
     * Constructs a dataset provenance using the supplied information.
     * @param sourceProvenance The data source provenance.
     * @param transformationProvenance The transformations applied to the dataset.
     * @param datasetClassName The dataset class name.
     * @param isDense Is the dataset dense?
     * @param isSequence Is it a sequence dataset?
     * @param numExamples The number of examples in this dataset.
     * @param numFeatures The number of features in this dataset.
     * @param numOutputs The output dimensionality.
     */
    protected DatasetProvenance(DataProvenance sourceProvenance, ListProvenance<ObjectProvenance> transformationProvenance, String datasetClassName, boolean isDense, boolean isSequence, int numExamples, int numFeatures, int numOutputs) {
        this.className = datasetClassName;
        this.sourceProvenance = sourceProvenance;
        this.transformationProvenance = transformationProvenance;
        this.isDense = isDense;
        this.isSequence = isSequence;
        this.numExamples = numExamples;
        this.numFeatures = numFeatures;
        this.numOutputs = numOutputs;
        this.versionString = Tribuo.VERSION;
    }

    /**
     * Deserialization constructor.
     * @param map The provenances.
     */
    @SuppressWarnings("unchecked") //ListProvenance assignment
    public DatasetProvenance(Map<String,Provenance> map) {
        this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
        this.sourceProvenance = ObjectProvenance.checkAndExtractProvenance(map,DATASOURCE,DataProvenance.class, DatasetProvenance.class.getSimpleName());
        this.transformationProvenance = ObjectProvenance.checkAndExtractProvenance(map,TRANSFORMATIONS,ListProvenance.class, DatasetProvenance.class.getSimpleName());
        this.isDense = ObjectProvenance.checkAndExtractProvenance(map,IS_DENSE,BooleanProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
        this.isSequence = ObjectProvenance.checkAndExtractProvenance(map,IS_SEQUENCE,BooleanProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
        this.numExamples = ObjectProvenance.checkAndExtractProvenance(map,NUM_EXAMPLES,IntProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
        this.numFeatures = ObjectProvenance.checkAndExtractProvenance(map,NUM_FEATURES,IntProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
        this.numOutputs = ObjectProvenance.checkAndExtractProvenance(map,NUM_OUTPUTS,IntProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
        this.versionString = ObjectProvenance.checkAndExtractProvenance(map, TRIBUO_VERSION_STRING,StringProvenance.class, DatasetProvenance.class.getSimpleName()).getValue();
    }

    @Override
    public String getClassName() {
        return className;
    }

    /**
     * The input data provenance.
     * @return The data provenance.
     */
    public DataProvenance getSourceProvenance() {
        return sourceProvenance;
    }

    /**
     * The transformation provenances, in application order.
     * @return The transformation provenances.
     */
    public ListProvenance<ObjectProvenance> getTransformationProvenance() {
        return transformationProvenance;
    }

    /**
     * Is the Dataset dense?
     * @return True if dense.
     */
    public boolean isDense() {
        return isDense;
    }

    /**
     * Is it a sequence dataset?
     * @return True if a sequence dataset.
     */
    public boolean isSequence() {
        return isSequence;
    }

    /**
     * The number of examples.
     * @return The number of examples.
     */
    public int getNumExamples() {
        return numExamples;
    }

    /**
     * The number of features.
     * @return The number of features.
     */
    public int getNumFeatures() {
        return numFeatures;
    }

    /**
     * The number of output dimensions.
     * @return The number of output dimensions.
     */
    public int getNumOutputs() {
        return numOutputs;
    }

    /**
     * The Tribuo version used to create this dataset.
     * @return The Tribuo version.
     */
    public String getTribuoVersion() {
        return versionString;
    }

    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        List<Pair<String,Provenance>> iterable = allProvenances();
        return iterable.iterator();
    }

    /**
     * Returns a list of all the provenances.
     * @return The provenances.
     */
    protected List<Pair<String,Provenance>> allProvenances() {
        ArrayList<Pair<String,Provenance>> provenances = new ArrayList<>();
        provenances.add(new Pair<>(CLASS_NAME,new StringProvenance(CLASS_NAME,className)));
        provenances.add(new Pair<>(DATASOURCE,sourceProvenance));
        provenances.add(new Pair<>(TRANSFORMATIONS,transformationProvenance));
        provenances.add(new Pair<>(IS_SEQUENCE,new BooleanProvenance(IS_SEQUENCE,isSequence)));
        provenances.add(new Pair<>(IS_DENSE,new BooleanProvenance(IS_DENSE,isDense)));
        provenances.add(new Pair<>(NUM_EXAMPLES,new IntProvenance(NUM_EXAMPLES,numExamples)));
        provenances.add(new Pair<>(NUM_FEATURES,new IntProvenance(NUM_FEATURES,numFeatures)));
        provenances.add(new Pair<>(NUM_OUTPUTS,new IntProvenance(NUM_OUTPUTS,numOutputs)));
        provenances.add(new Pair<>(TRIBUO_VERSION_STRING,new StringProvenance(TRIBUO_VERSION_STRING,versionString)));
        return provenances;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DatasetProvenance)) return false;
        DatasetProvenance pairs = (DatasetProvenance) o;
        return isDense == pairs.isDense &&
                isSequence == pairs.isSequence &&
                numExamples == pairs.numExamples &&
                numFeatures == pairs.numFeatures &&
                numOutputs == pairs.numOutputs &&
                className.equals(pairs.className) &&
                sourceProvenance.equals(pairs.sourceProvenance) &&
                transformationProvenance.equals(pairs.transformationProvenance) &&
                versionString.equals(pairs.versionString);
    }

    @Override
    public int hashCode() {
        return Objects.hash(className, sourceProvenance, transformationProvenance, isDense, isSequence, numExamples, numFeatures, numOutputs, versionString);
    }

    @Override
    public String toString() {
        if (isSequence) {
            return generateString("SequenceDataset");
        } else {
            return generateString("Dataset");
        }
    }
}
