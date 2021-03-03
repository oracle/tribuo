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

package org.tribuo;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.transform.TransformerMap;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A MutableDataset is a {@link Dataset} with a {@link MutableFeatureMap} which grows over time.
 * Whenever an {@link Example} is added to the dataset it observes each feature and output
 * keeping appropriate statistics in the {@link FeatureMap} and {@link OutputInfo}.
 */
public class MutableDataset<T extends Output<T>> extends Dataset<T> {
    
    private static final Logger logger = Logger.getLogger(MutableDataset.class.getName());
    
    private static final long serialVersionUID = 1L;

    /**
     * Information about the outputs in this dataset.
     */
    protected final MutableOutputInfo<T> outputMap;

    /**
     * A map from feature names to feature info objects.
     */
    protected final MutableFeatureMap featureMap;

    /**
     * The provenances of the transformations applied to this dataset.
     */
    protected final List<ObjectProvenance> transformProvenances = new ArrayList<>();

    /**
     * Denotes if this dataset contains implicit zeros or not.
     */
    protected boolean dense = true;

    /**
     * Creates an empty dataset.
     * @param sourceProvenance A description of the input data, including preprocessing steps.
     * @param outputFactory The output factory.
     */
    public MutableDataset(DataProvenance sourceProvenance, OutputFactory<T> outputFactory) {
        super(sourceProvenance,outputFactory);
        this.featureMap = new MutableFeatureMap();
        this.outputMap = outputFactory.generateInfo();
    }

    /**
     * Creates a dataset from a data source. This method will create the output
     * and feature maps that are needed for training and evaluating classifiers.
     * @param dataSource The examples.
     * @param provenance A description of the input data, including preprocessing steps.
     * @param outputFactory The output factory.
     */
    public MutableDataset(Iterable<Example<T>> dataSource, DataProvenance provenance, OutputFactory<T> outputFactory) {
        super(provenance,outputFactory);
        this.featureMap = new MutableFeatureMap();
        this.outputMap = outputFactory.generateInfo();
        for (Example<T> ex : dataSource) {
            add(ex);
        }
    }

    /**
     * Creates a dataset from a data source. This method creates the output and feature maps
     * needed for training and evaluating classifiers.
     * @param dataSource The examples.
     */
    public MutableDataset(DataSource<T> dataSource) {
        this(dataSource,dataSource.getProvenance(),dataSource.getOutputFactory());
    }

    /**
     * Adds an example to the dataset, which observes the output and each feature value.
     * <p>
     * It also canonicalises the reference to each feature's name (i.e., replacing the reference
     * to a feature's name with the canonical one stored in this Dataset's {@link VariableInfo}).
     * This greatly reduces the memory footprint.
     * @param ex The example to add.
     */
    public void add(Example<T> ex) {
        if (!ex.validateExample()) {
            throw new IllegalArgumentException("Example had duplicate features, invalid features or no features.");
        }
        outputMap.observe(ex.getOutput());
        data.add(ex);
        int oldNumFeatures = featureMap.size();
        for (Feature f : ex) {
            featureMap.add(f.getName(),f.getValue());
        }
        ex.canonicalize(featureMap);
        // If we've observed a new feature, or this example doesn't contain all the features then
        // the dataset stops being dense.
        if ((oldNumFeatures != 0) && (oldNumFeatures < featureMap.size() || ex.size() != featureMap.size())) {
            dense = false;
        }
    }

    /**
     * Adds all the Examples in the supplied collection to this dataset.
     * @param collection The collection of Examples.
     */
    public void addAll(Collection<? extends Example<T>> collection) {
        for (Example<T> e : collection) {
            add(e);
        }
    }

    /**
     * Sets the weights in each example according to their output.
     * @param weights A map of {@link Output}s to float weights.
     */
    public void setWeights(Map<T,Float> weights) {
        for (Example<T> e : this) {
            Float weight = weights.get(e.getOutput());
            if (weight != null) {
                e.setWeight(weight);
            } else {
                e.setWeight(1.0f);
            }
        }
    }

    /**
     * Gets the set of possible outputs in this dataset.
     * <p>
     * In the case of regression returns a Set containing dimension names.
     * @return The set of possible outputs.
     */
    @Override
    public Set<T> getOutputs() {
        return outputMap.getDomain();
    }

    @Override
    public ImmutableFeatureMap getFeatureIDMap() {
        return new ImmutableFeatureMap(featureMap);
    }

    @Override
    public MutableFeatureMap getFeatureMap() {
        return featureMap;
    }

    @Override
    public ImmutableOutputInfo<T> getOutputIDInfo() {
        return outputMap.generateImmutableOutputInfo();
    }

    @Override
    public OutputInfo<T> getOutputInfo() {
        return outputMap;
    }

    @Override
    public String toString(){
        if (transformProvenances.isEmpty()) {
            return "MutableDataset(source=" + sourceProvenance + ",isDense="+dense+")";
        } else {
            return "MutableDataset(source=" + sourceProvenance + ",isDense="+dense+",transforms="+transformProvenances.toString()+")";
        }
    }

    /**
     * Is the dataset dense (i.e., do all features in the domain have a value in each example).
     * @return True if the dataset is dense.
     */
    public boolean isDense() {
        return dense;
    }

    /**
     * Applies all the transformations from the {@link TransformerMap} to this dataset.
     * @param transformerMap The transformations to apply.
     */
    public void transform(TransformerMap transformerMap) {
        featureMap.clear();
        logger.fine(String.format("Transforming %,d examples", data.size()));
        int nt = 0;
        for (Example<T> example : data) {
            example.transform(transformerMap);
            for (Feature f : example) {
                featureMap.add(f.getName(),f.getValue());
            }
            nt++;
            if(logger.isLoggable(Level.FINE) && nt % 10000 == 0) {
                logger.fine(String.format("Transformed %,d/%,d", nt, data.size()));
            }
        }
        transformProvenances.add(transformerMap.getProvenance());
    }

    /**
     * Iterates through the examples, converting implicit zeros into explicit zeros.
     */
    public void densify() {
        ArrayList<String> featureNames = new ArrayList<>(featureMap.keySet());
        Collections.sort(featureNames);
        for (Example<T> example : data) {
            example.densify(featureNames);
        }
        dense = true;
    }

    /**
     * Clears all the examples out of this dataset, and flushes the FeatureMap, OutputInfo, and transform provenances.
     */
    public void clear() {
        outputMap.clear();
        featureMap.clear();
        data.clear();
        transformProvenances.clear();
        dense = true;
    }

    @Override
    public DatasetProvenance getProvenance() {
        return new DatasetProvenance(sourceProvenance, new ListProvenance<>(transformProvenances), this);
    }

    /**
     * Creates a deep copy of the supplied {@link Dataset} which is mutable.
     * <p>
     * Copies the individual examples using their copy method.
     * @param other The dataset to copy.
     * @param <T> The output type.
     * @return A mutable deep copy of the dataset.
     */
    public static <T extends Output<T>> MutableDataset<T> createDeepCopy(Dataset<T> other) {
        MutableDataset<T> copy = new MutableDataset<>(other.getProvenance(),other.outputFactory);

        for (Example<T> e : other) {
            copy.add(e.copy());
        }

        return copy;
    }
}
