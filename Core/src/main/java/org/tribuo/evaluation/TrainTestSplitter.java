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

package org.tribuo.evaluation;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.BooleanProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DoubleProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.IntProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.provenance.DataSourceProvenance;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * Splits data into training and testing sets. Note that this doesn't
 * operate on {@link org.tribuo.Dataset}, but rather on {@link DataSource}.
 * @param <T> The output type of the examples in the datasource.
 */
public class TrainTestSplitter<T extends Output<T>> {

    private final DataSource<T> train;
    
    private final DataSource<T> test;

    private final DataSourceProvenance originalProvenance;

    private final long seed;

    private final double trainProportion;

    private final int size;

    /**
     * Creates a splitter that splits a dataset 70/30 train and test using a default seed.
     * @param data The data to split.
     */
    public TrainTestSplitter(DataSource<T> data) {
        this(data,1);
    }

    /**
     * Creates a splitter that splits a dataset 70/30 train and test.
     * @param data The data to split.
     * @param seed The seed for the RNG.
     */
    public TrainTestSplitter(DataSource<T> data, long seed) {
        this(data, 0.7, seed);
    }
    
    /**
     * Creates a splitter that will split the given data set into 
     * a training and testing set. The give proportion of the data will be
     * randomly selected for the training set. The remainder will be in the 
     * test set.
     * 
     * @param data the data that we want to split.
     * @param trainProportion the proportion of the data to select for training. 
     * This should be a number between 0 and 1. For example, a value of 0.7 means
     * that 70% of the data should be selected for the training set.
     * @param seed The seed for the RNG.
     */
    public TrainTestSplitter(DataSource<T> data, double trainProportion, long seed) {
        this.seed = seed;
        this.trainProportion = trainProportion;
        this.originalProvenance = data.getProvenance();
        List<Example<T>> l = new ArrayList<>();
        for(Example<T> ex : data) {
            l.add(ex);
        }
        this.size = l.size();
        Random rng = new Random(seed);
        Collections.shuffle(l,rng);
        int n = (int) (trainProportion * l.size());
        train = new ListDataSource<>(l.subList(0, n),data.getOutputFactory(),new SplitDataSourceProvenance(this,true));
        test = new ListDataSource<>(l.subList(n, l.size()),data.getOutputFactory(),new SplitDataSourceProvenance(this,false));
    }

    /**
     * The total amount of data in train and test combined.
     * @return The number of examples.
     */
    public int totalSize() {
        return size;
    }

    /**
     * Gets the training data source.
     * @return The training data.
     */
    public DataSource<T> getTrain() {
        return train;
    }

    /**
     * Gets the testing datasource.
     * @return The testing data.
     */
    public DataSource<T> getTest() {
        return test;
    }

    /**
     * Provenance for a split data source.
     */
    public static class SplitDataSourceProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private static final String SOURCE = "source";
        private static final String TRAIN_PROPORTION = "train-proportion";
        private static final String SEED = "seed";
        private static final String SIZE = "size";
        private static final String IS_TRAIN = "is-train";

        private final StringProvenance className;
        private final DataSourceProvenance innerSourceProvenance;
        private final DoubleProvenance trainProportion;
        private final LongProvenance seed;
        private final IntProvenance size;
        private final BooleanProvenance isTrain;

        <T extends Output<T>> SplitDataSourceProvenance(TrainTestSplitter<T> host, boolean isTrain) {
            this.className = new StringProvenance(CLASS_NAME,host.getClass().getName());
            this.innerSourceProvenance = host.originalProvenance;
            this.trainProportion = new DoubleProvenance(TRAIN_PROPORTION,host.trainProportion);
            this.seed = new LongProvenance(SEED,host.seed);
            this.size = new IntProvenance(SIZE,host.size);
            this.isTrain = new BooleanProvenance(IS_TRAIN,isTrain);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public SplitDataSourceProvenance(Map<String, Provenance> map) {
            this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class,SplitDataSourceProvenance.class.getSimpleName());
            this.innerSourceProvenance = ObjectProvenance.checkAndExtractProvenance(map,SOURCE,DataSourceProvenance.class,SplitDataSourceProvenance.class.getSimpleName());
            this.trainProportion = ObjectProvenance.checkAndExtractProvenance(map,TRAIN_PROPORTION,DoubleProvenance.class,SplitDataSourceProvenance.class.getSimpleName());
            this.seed = ObjectProvenance.checkAndExtractProvenance(map,SEED,LongProvenance.class,SplitDataSourceProvenance.class.getSimpleName());
            this.size = ObjectProvenance.checkAndExtractProvenance(map,SIZE,IntProvenance.class,SplitDataSourceProvenance.class.getSimpleName());
            this.isTrain = ObjectProvenance.checkAndExtractProvenance(map,IS_TRAIN,BooleanProvenance.class,SplitDataSourceProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return className.getValue();
        }

        @Override
        public Iterator<Pair<String, Provenance>> iterator() {
            ArrayList<Pair<String,Provenance>> list = new ArrayList<>();

            list.add(new Pair<>(CLASS_NAME,className));
            list.add(new Pair<>(SOURCE,innerSourceProvenance));
            list.add(new Pair<>(TRAIN_PROPORTION,trainProportion));
            list.add(new Pair<>(SEED,seed));
            list.add(new Pair<>(SIZE,size));
            list.add(new Pair<>(IS_TRAIN,isTrain));

            return list.iterator();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof SplitDataSourceProvenance)) return false;
            SplitDataSourceProvenance pairs = (SplitDataSourceProvenance) o;
            return className.equals(pairs.className) &&
                    innerSourceProvenance.equals(pairs.innerSourceProvenance) &&
                    trainProportion.equals(pairs.trainProportion) &&
                    seed.equals(pairs.seed) &&
                    size.equals(pairs.size) &&
                    isTrain.equals(pairs.isTrain);
        }

        @Override
        public int hashCode() {
            return Objects.hash(className, innerSourceProvenance, trainProportion, seed, size, isTrain);
        }

        @Override
        public String toString() {
            return "SplitDataSourceProvenance(" +
                    "className=" + className +
                    ",innerSourceProvenance=" + innerSourceProvenance +
                    ",trainProportion=" + trainProportion +
                    ",seed=" + seed +
                    ",size=" + size +
                    ",isTrain=" + isTrain +
                    ')';
        }
    }
}
