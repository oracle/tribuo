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

package org.tribuo.common.xgboost;

import com.oracle.labs.mlrg.olcut.util.MutableDouble;
import com.oracle.labs.mlrg.olcut.util.Pair;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.common.xgboost.XGBoostTrainer.DMatrixTuple;
import org.tribuo.provenance.ModelProvenance;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.PriorityQueue;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps around a XGBoost.Booster.
 * <p>
 * XGBoost is a fast implementation of gradient boosted decision trees.
 * <p>
 * Throws IllegalStateException if the XGBoost C++ library fails to load or throws an exception.
 * <p>
 * See:
 * <pre>
 * Chen T, Guestrin C.
 * "XGBoost: A Scalable Tree Boosting System"
 * Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Friedman JH.
 * "Greedy Function Approximation: a Gradient Boosting Machine"
 * Annals of statistics, 2001.
 * </pre>
 * <p>
 * Note: XGBoost requires a native library, on macOS this library requires libomp (which can be installed via homebrew),
 * on Windows this native library must be compiled into a jar as it's not contained in the official XGBoost binary
 * on Maven Central.
 */
public final class XGBoostModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 4L;

    private static final Logger logger = Logger.getLogger(XGBoostModel.class.getName());

    private final XGBoostOutputConverter<T> converter;

    /**
     * The XGBoost4J Boosters.
     */
    protected transient List<Booster> models;

    XGBoostModel(String name, ModelProvenance description,
                 ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> labelIDMap,
                 List<Booster> models, XGBoostOutputConverter<T> converter) {
        super(name,description,featureIDMap,labelIDMap,converter.generatesProbabilities());
        this.converter = converter;
        this.models = models;
    }

    /**
     * Returns an unmodifiable list containing a copy of each model.
     * <p>
     * As XGBoost4J models don't expose a copy constructor this requires
     * serializing each model to a byte array and rebuilding it, and is thus quite expensive.
     * @return A copy of all of the models.
     */
    public List<Booster> getInnerModels() {
        List<Booster> copy = new ArrayList<>();

        for (Booster m : models) {
            copy.add(copyModel(m));
        }

        return Collections.unmodifiableList(copy);
    }

    /**
     * Sets the number of threads to use at prediction time.
     * <p>
     * If set to 0 sets nthreads = num hardware threads.
     * @param threads The new number of threads.
     */
    public void setNumThreads(int threads) {
        if (threads > -1) {
            try {
                for (Booster model : models) {
                    model.setParam("nthread", threads);
                }
            } catch (XGBoostError e) {
                logger.log(Level.SEVERE, "XGBoost threw an error", e);
                throw new IllegalStateException(e);
            }
        }
    }

    /**
     * Uses the model to predict the labels for multiple examples contained in
     * a data set.
     * @param examples the data set containing the examples to predict.
     * @return the results of the predictions, in the same order as the
     * data set generates the example.
     */
    @Override
    public List<Prediction<T>> predict(Dataset<T> examples) {
        return predict(examples.getData());
    }

    /**
     * Uses the model to predict the label for multiple examples.
     * @param examples the examples to predict.
     * @return the results of the prediction, in the same order as the
     * examples.
     */
    @Override
    public List<Prediction<T>> predict(Iterable<Example<T>> examples) {
        try {
            DMatrixTuple<T> testMatrix = XGBoostTrainer.convertExamples(examples,featureIDMap);
            List<float[][]> outputs = new ArrayList<>();
            for (Booster model : models) {
                outputs.add(model.predict(testMatrix.data));
            }

            int[] numValidFeatures = testMatrix.numValidFeatures;
            Example<T>[] exampleArray = testMatrix.examples;
            return converter.convertBatchOutput(outputIDInfo,outputs,numValidFeatures,exampleArray);
        } catch (XGBoostError e) {
            logger.log(Level.SEVERE, "XGBoost threw an error", e);
            throw new IllegalStateException(e);
        }

    }

    @Override
    public Prediction<T> predict(Example<T> example) {
        try {
            DMatrixTuple<T> testData = XGBoostTrainer.convertExample(example,featureIDMap);
            List<float[]> outputs = new ArrayList<>();
            for (Booster model : models) {
                outputs.add(model.predict(testData.data)[0]);
            }
            Prediction<T> pred = converter.convertOutput(outputIDInfo,outputs,testData.numValidFeatures[0],example);
            return pred;
        } catch (XGBoostError e) {
            logger.log(Level.SEVERE, "XGBoost threw an error", e);
            throw new IllegalStateException(e);
        }
    }

    @Override
    public Map<String, List<Pair<String,Double>>> getTopFeatures(int n) {
        try {
            int maxFeatures = n < 0 ? featureIDMap.size() : n;
            // Aggregate feature scores across all the models.
            // This throws away model specific information which is useful in the case of regression,
            // but it's very tricky to get the dimension name associated with the model.
            Map<String, MutableDouble> outputMap = new HashMap<>();
            for (Booster model : models) {
                Map<String, Integer> xgboostMap = model.getFeatureScore("");
                for (Map.Entry<String,Integer> f : xgboostMap.entrySet()) {
                    int id = Integer.parseInt(f.getKey().substring(1));
                    String name = featureIDMap.get(id).getName();
                    MutableDouble curVal = outputMap.computeIfAbsent(name,(k)->new MutableDouble());
                    curVal.increment(f.getValue());
                }
            }
            Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
            PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures,comparator);
            for (Map.Entry<String,MutableDouble> e : outputMap.entrySet()) {
                Pair<String,Double> cur = new Pair<>(e.getKey(), e.getValue().doubleValue());

                if (q.size() < maxFeatures) {
                    q.offer(cur);
                } else if (comparator.compare(cur,q.peek()) > 0) {
                    q.poll();
                    q.offer(cur);
                }
            }
            List<Pair<String,Double>> list = new ArrayList<>();
            while(q.size() > 0) {
                list.add(q.poll());
            }
            Collections.reverse(list);

            Map<String, List<Pair<String,Double>>> map = new HashMap<>();
            map.put(Model.ALL_OUTPUTS,list);

            return map;
        } catch (XGBoostError e) {
            logger.log(Level.SEVERE, "XGBoost threw an error", e);
            return Collections.emptyMap();
        }
    }

    /**
     * Returns the string model dumps from each Booster.
     * @return The model dumps.
     */
    public List<String[]> getModelDump() {
        try {
            List<String[]> list = new ArrayList<>();
            for (Booster m : models) {
                list.add(m.getModelDump("", true));
            }
            return list;
        } catch (XGBoostError e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public Optional<Excuse<T>> getExcuse(Example<T> example) {
        return Optional.empty();
    }

    /**
     * Copies a single XGBoost Booster by serializing and deserializing it.
     * @param booster The booster to copy.
     * @return A deep copy of the booster.
     */
    static Booster copyModel(Booster booster) {
        try {
            byte[] serialisedBooster = booster.toByteArray();
            return XGBoost.loadModel(new ByteArrayInputStream(serialisedBooster));
        } catch (XGBoostError | IOException e) {
            throw new IllegalStateException("Unable to copy XGBoost model.",e);
        }
    }

    @Override
    protected Model<T> copy(String newName, ModelProvenance newProvenance) {
        List<Booster> newModels = new ArrayList<>();
        for (Booster model : models) {
            newModels.add(copyModel(model));
        }
        return new XGBoostModel<>(newName, newProvenance, featureIDMap, outputIDInfo, newModels, converter);
    }

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        try {
            out.writeInt(models.size());
            for (Booster model : models) {
                byte[] serialisedBooster = model.toByteArray();
                out.writeObject(serialisedBooster);
            }
        } catch (XGBoostError e) {
            throw new IOException("Failed to serialize the XGBoost model",e);
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        try {
            models = new ArrayList<>();
            int numModels = in.readInt();
            for (int i = 0; i < numModels; i++) {
                // Now read in the byte array and rebuild each Booster
                byte[] serialisedBooster = (byte[]) in.readObject();
                models.add(XGBoost.loadModel(new ByteArrayInputStream(serialisedBooster)));
            }
        } catch (XGBoostError e) {
            throw new IOException("Failed to deserialize the XGBoost model",e);
        }
    }
}
