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

package org.tribuo.sequence;

import com.oracle.labs.mlrg.olcut.provenance.ListProvenance;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableFeatureMap;
import org.tribuo.MutableOutputInfo;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.OutputInfo;
import org.tribuo.provenance.DataProvenance;
import org.tribuo.provenance.DatasetProvenance;

import java.util.Collection;
import java.util.Set;

/**
 * A MutableSequenceDataset is a {@link SequenceDataset} with a {@link MutableFeatureMap} which grows over time.
 * Whenever an {@link SequenceExample} is added to the dataset.
 */
public class MutableSequenceDataset<T extends Output<T>> extends SequenceDataset<T> {
    private static final long serialVersionUID = 1L;

    /**
     * A map from labels to IDs for the labels found in this dataset.
     */
    protected final MutableOutputInfo<T> outputInfo;

    /**
     * A map from feature names to IDs for the features found in this dataset.
     */
    protected final MutableFeatureMap featureMap;

    /**
     * Does this dataset have a dense feature space.
     */
    protected boolean dense = true;

    /**
     * Creates an empty sequence dataset.
     * @param sourceProvenance A description of the input data, including preprocessing steps.
     * @param outputFactory The output factory.
     */
    public MutableSequenceDataset(DataProvenance sourceProvenance, OutputFactory<T> outputFactory) {
        super(sourceProvenance, outputFactory);
        this.featureMap = new MutableFeatureMap();
        this.outputInfo = outputFactory.generateInfo();
    }

    /**
     * Creates a dataset from a data source. This method will create the output
     * and feature ID maps that are needed for training and evaluating classifiers.
     * @param dataSource The input data.
     * @param sourceProvenance A description of the data, including preprocessing steps.
     * @param outputFactory The output factory.
     */
    public MutableSequenceDataset(Iterable<SequenceExample<T>> dataSource, DataProvenance sourceProvenance, OutputFactory<T> outputFactory) {
        super(sourceProvenance, outputFactory);
        this.featureMap = new MutableFeatureMap();
        this.outputInfo = outputFactory.generateInfo();
        for (SequenceExample<T> ex : dataSource) {
            add(ex);
        }
    }

    /**
     * Builds a dataset from the supplied data source.
     * @param dataSource The data source.
     */
    public MutableSequenceDataset(SequenceDataSource<T> dataSource) {
        this(dataSource,dataSource.getProvenance(),dataSource.getOutputFactory());
    }

    /**
     * Copies the immutable dataset into a mutable dataset.
     * <p>
     * This should be infrequently used and mostly exists for the ViterbiTrainer.
     * @param dataset The dataset to copy.
     */
    //special purpose constructor created for ViterbiTrainer
    public MutableSequenceDataset(ImmutableSequenceDataset<T> dataset) {
        super(dataset.getProvenance(),dataset.getOutputFactory());

        this.featureMap = new MutableFeatureMap();
        this.outputInfo = dataset.getOutputInfo().generateMutableOutputInfo();
        for (SequenceExample<T> ex : dataset) {
            add(new SequenceExample<>(ex));
        }
    }

    /**
     * Clears all the examples out of this dataset, and flushes the FeatureMap, OutputInfo, and transform provenances.
     */
    public void clear() {
        outputInfo.clear();
        featureMap.clear();
        data.clear();
        dense = true;
    }

    /**
     * Adds a {@link SequenceExample} to this dataset.
     * <p>
     * It also canonicalises the reference to each feature's name (i.e., replacing the reference
     * to a feature's name with the canonical one stored in this Dataset's {@link org.tribuo.VariableInfo}).
     * This greatly reduces the memory footprint.
     * @param ex The example to add.
     */
    public void add(SequenceExample<T> ex) {
        if (!ex.validateExample()) {
            throw new IllegalArgumentException("SequenceExample had duplicate features, no features or no Examples.");
        }
        data.add(ex);
        int oldNumFeatures = featureMap.size();
        boolean exampleIsDense = true;
        for (Example<T> e : ex) {
            outputInfo.observe(e.getOutput());
            for (Feature f : e) {
                featureMap.add(f.getName(),f.getValue());
            }
            if (e.size() != featureMap.size()) {
                exampleIsDense = false;
            }
        }
        ex.canonicalise(featureMap);
        // If we've observed a new feature, or this example doesn't contain all the features then
        // the dataset stops being dense.
        if (((oldNumFeatures != 0) && (oldNumFeatures < featureMap.size())) || !exampleIsDense) {
            dense = false;
        }
    }

    /**
     * Adds all the SequenceExamples in the supplied collection to this dataset.
     * @param collection The collection of SequenceExamples.
     */
    public void addAll(Collection<SequenceExample<T>> collection) {
        for (SequenceExample<T> e : collection) {
            add(e);
        }
    }

    @Override
    public Set<T> getOutputs() {
        return outputInfo.getDomain();
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
        return outputInfo.generateImmutableOutputInfo();
    }

    @Override
    public OutputInfo<T> getOutputInfo() {
        return outputInfo;
    }

    /**
     * Is the dataset dense (i.e., do all features in the domain have a value in each example).
     * @return True if the dataset is dense.
     */
    public boolean isDense() {
        return dense;
    }

    /**
     * Iterates through the examples, converting implicit zeros into explicit zeros.
     */
    public void densify() {
        for (SequenceExample<T> example : data) {
            example.densify(featureMap);
        }
        dense = true;
    }

    @Override
    public String toString(){
        return "MutableSequenceDataset(source="+ sourceProvenance.toString()+")";
    }

    @Override
    public DatasetProvenance getProvenance() {
        return new DatasetProvenance(sourceProvenance, new ListProvenance<>(), this);
    }
}
