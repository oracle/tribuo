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

import com.oracle.labs.mlrg.olcut.util.Pair;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.interop.ExternalDatasetProvenance;
import org.tribuo.interop.ExternalModel;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.math.la.SparseVector;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps around a XGBoost.Booster which was trained by a system other than Tribuo.
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
 * <p>
 * and for the original algorithm:
 * <pre>
 * Friedman JH.
 * "Greedy Function Approximation: a Gradient Boosting Machine"
 * Annals of statistics, 2001.
 * </pre>
 * <p>
 * N.B.: XGBoost4J wraps the native C implementation of xgboost that links to various C libraries, including libgomp
 * and glibc (on Linux). If you're running on Alpine, which does not natively use glibc, you'll need to install glibc
 * into the container.
 * On the macOS binary on Maven Central is compiled without
 * OpenMP support, meaning that XGBoost is single threaded on macOS. You can recompile the macOS binary with
 * OpenMP support after installing libomp from homebrew if necessary.
 */
public final class XGBoostExternalModel<T extends Output<T>> extends ExternalModel<T,DMatrix,float[][]> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(XGBoostExternalModel.class.getName());

    private final XGBoostOutputConverter<T> converter;

    /**
     * Transient as we rely upon the native serialisation mechanism to bytes rather than Java serializing the Booster.
     */
    protected transient Booster model;

    private XGBoostExternalModel(String name, ModelProvenance provenance,
                                 ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                 Map<String, Integer> featureMapping, Booster model,
                                 XGBoostOutputConverter<T> converter) {
        super(name, provenance, featureIDMap, outputIDInfo, converter.generatesProbabilities(), featureMapping);
        this.model = model;
        this.converter = converter;
    }

    private XGBoostExternalModel(String name, ModelProvenance provenance,
                                 ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                 int[] featureForwardMapping, int[] featureBackwardMapping,
                                 Booster model, XGBoostOutputConverter<T> converter) {
        super(name,provenance,featureIDMap,outputIDInfo,featureForwardMapping,featureBackwardMapping,
              converter.generatesProbabilities());
        this.model = model;
        this.converter = converter;
    }

    @Override
    protected DMatrix convertFeatures(SparseVector input) {
        try {
            return XGBoostTrainer.convertSparseVector(input);
        } catch (XGBoostError e) {
            logger.severe("XGBoost threw an error while constructing the DMatrix.");
            throw new IllegalStateException(e);
        }
    }

    @Override
    protected DMatrix convertFeaturesList(List<SparseVector> input) {
        try {
            return XGBoostTrainer.convertSparseVectors(input);
        } catch (XGBoostError e) {
            logger.severe("XGBoost threw an error while constructing the DMatrix.");
            throw new IllegalStateException(e);
        }
    }

    @Override
    protected float[][] externalPrediction(DMatrix input) {
        try {
            return model.predict(input);
        } catch (XGBoostError e) {
            logger.severe("XGBoost threw an error while predicting.");
            throw new IllegalStateException(e);
        }
    }

    @Override
    protected Prediction<T> convertOutput(float[][] output, int numValidFeatures, Example<T> example) {
        return converter.convertOutput(outputIDInfo,Collections.singletonList(output[0]),numValidFeatures,example);
    }

    @SuppressWarnings("unchecked") // generic array creation
    @Override
    protected List<Prediction<T>> convertOutput(float[][] output, int[] numValidFeatures, List<Example<T>> examples) {
        return converter.convertBatchOutput(outputIDInfo,Collections.singletonList(output),numValidFeatures,(Example<T>[])examples.toArray(new Example[0]));
    }

    /**
     * Creates objects to report feature importance metrics for XGBoost. See the documentation of {@link XGBoostFeatureImportance}
     * for more information on what those metrics mean. Typically this list will contain a single instance for the entire
     * model. For multidimensional regression the list will have one entry per dimension, in dimension order.
     * @return The feature importance object(s).
     */
    public List<XGBoostFeatureImportance> getFeatureImportance() {
        return Collections.singletonList(new XGBoostFeatureImportance(model, this));
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        try {
            int maxFeatures = n < 0 ? featureIDMap.size() : n;
            Map<String, Integer> xgboostMap = model.getFeatureScore("");
            Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
            PriorityQueue<Pair<String,Double>> q = new PriorityQueue<>(maxFeatures,comparator);
            //iterate over the scored features
            for (Map.Entry<String, Integer> f : xgboostMap.entrySet()) {
                int id = Integer.parseInt(f.getKey().substring(1));
                Pair<String,Double> cur = new Pair<>(featureIDMap.get(featureBackwardMapping[id]).getName(), (double) f.getValue());

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

    @Override
    protected XGBoostExternalModel<T> copy(String newName, ModelProvenance newProvenance) {
        return new XGBoostExternalModel<>(newName, newProvenance, featureIDMap, outputIDInfo,
                                          featureForwardMapping, featureBackwardMapping,
                                          XGBoostModel.copyModel(model), converter);
    }

    /**
     * Creates an {@code XGBoostExternalModel} from the supplied model on disk.
     * @param factory The output factory to use.
     * @param featureMapping The feature mapping between Tribuo names and XGBoost integer ids.
     * @param outputMapping The output mapping between Tribuo outputs and XGBoost integer ids.
     * @param outputFunc The XGBoostOutputConverter function for the output type.
     * @param path The path to the model on disk.
     * @param <T> The type of the output.
     * @return An XGBoostExternalModel ready to score new inputs.
     */
    public static <T extends Output<T>> XGBoostExternalModel<T> createXGBoostModel(OutputFactory<T> factory, Map<String, Integer> featureMapping, Map<T,Integer> outputMapping, XGBoostOutputConverter<T> outputFunc, String path) {
        try {
            Booster model = XGBoost.loadModel(path);
            return createXGBoostModel(factory,featureMapping,outputMapping,outputFunc,model,new File(path).toURI().toURL());
        } catch (XGBoostError | MalformedURLException e) {
            throw new IllegalArgumentException("Unable to load model from path " + path, e);
        }
    }

    /**
     * Creates an {@code XGBoostExternalModel} from the supplied model on disk.
     * @param factory The output factory to use.
     * @param featureMapping The feature mapping between Tribuo names and XGBoost integer ids.
     * @param outputMapping The output mapping between Tribuo outputs and XGBoost integer ids.
     * @param outputFunc The XGBoostOutputConverter function for the output type.
     * @param path The path to the model on disk.
     * @param <T> The type of the output.
     * @return An XGBoostExternalModel ready to score new inputs.
     */
    public static <T extends Output<T>> XGBoostExternalModel<T> createXGBoostModel(OutputFactory<T> factory, Map<String, Integer> featureMapping, Map<T,Integer> outputMapping, XGBoostOutputConverter<T> outputFunc, Path path) {
        try {
            Booster model = XGBoost.loadModel(Files.newInputStream(path));
            return createXGBoostModel(factory,featureMapping,outputMapping,outputFunc,model,path.toUri().toURL());
        } catch (XGBoostError | IOException e) {
            throw new IllegalArgumentException("Unable to load model from path " + path, e);
        }
    }

    /**
     * Creates an {@code XGBoostExternalModel} from the supplied model.
     * <p>
     * Note: the provenance system requires that the URL point to a valid local file and
     * will throw an exception if it is not. However it doesn't check that the file is
     * where the Booster was created from.
     * We will replace this entry point with one that accepts useful provenance information
     * for an in-memory {@code Booster} object in a future release, and deprecate this
     * endpoint at that time.
     * @param factory The output factory to use.
     * @param featureMapping The feature mapping between Tribuo names and XGBoost integer ids.
     * @param outputMapping The output mapping between Tribuo outputs and XGBoost integer ids.
     * @param outputFunc The XGBoostOutputConverter function for the output type.
     * @param model The XGBoost model to wrap.
     * @param provenanceLocation The location where the model was loaded from.
     * @param <T> The type of the output.
     * @return An XGBoostExternalModel ready to score new inputs.
     */
    public static <T extends Output<T>> XGBoostExternalModel<T> createXGBoostModel(OutputFactory<T> factory, Map<String,Integer> featureMapping, Map<T,Integer> outputMapping, XGBoostOutputConverter<T> outputFunc, Booster model, URL provenanceLocation) {
        //TODO: add a new version of this method which accepts useful instance provenance information and deprecate this one
        ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
        ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory,outputMapping);
        OffsetDateTime now = OffsetDateTime.now();
        ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance(provenanceLocation);
        DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data",factory,false,featureMapping.size(),outputMapping.size());
        ModelProvenance provenance = new ModelProvenance(XGBoostExternalModel.class.getName(),now,datasetProvenance,trainerProvenance);
        return new XGBoostExternalModel<>("external-model",provenance,featureMap,outputInfo,
                                          featureMapping,model,outputFunc);
    }

    private void writeObject(ObjectOutputStream out) throws IOException {
        out.defaultWriteObject();
        try {
            byte[] serialisedBooster = model.toByteArray();
            out.writeObject(serialisedBooster);
        } catch (XGBoostError e) {
            throw new IOException("Failed to serialize the XGBoost model",e);
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        try {
            // Now read in the byte array and rebuild the Booster
            byte[] serialisedBooster = (byte[]) in.readObject();
            model = XGBoost.loadModel(new ByteArrayInputStream(serialisedBooster));
        } catch (XGBoostError e) {
            throw new IOException("Failed to deserialize the XGBoost model",e);
        }
    }
}
