/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
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
import org.tribuo.common.xgboost.protos.XGBoostModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.provenance.TrainerProvenance;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.InvocationTargetException;
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
import java.util.stream.Collectors;

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
 * N.B.: XGBoost4J wraps the native C implementation of xgboost that links to various C libraries, including libgomp
 * and glibc (on Linux). If you're running on Alpine, which does not natively use glibc, you'll need to install glibc
 * into the container.
 * On the macOS binary on Maven Central is compiled without
 * OpenMP support, meaning that XGBoost is single threaded on macOS. You can recompile the macOS binary with
 * OpenMP support after installing libomp from homebrew if necessary.
 */
public final class XGBoostModel<T extends Output<T>> extends Model<T> {
    private static final long serialVersionUID = 4L;

    private static final Logger logger = Logger.getLogger(XGBoostModel.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private final XGBoostOutputConverter<T> converter;

    // Used to signal if the model has been rewritten to fix the issue with multidimensional XGBoost regression models in 4.0 and 4.1.0.
    private boolean regression41MappingFix;

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
        this.regression41MappingFix = true;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @throws XGBoostError If the XGBoost byte array failed to parse.
     * @throws IOException If the XGBoost byte array failed to parse.
     * @return The deserialized object.
     */
    public static XGBoostModel<?> deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException, XGBoostError, IOException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        XGBoostModelProto proto = message.unpack(XGBoostModelProto.class);

        XGBoostOutputConverter<?> converter = ProtoUtil.deserialize(proto.getConverter());
        Class<?> converterWitness = converter.getTypeWitness();
        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(converterWitness)) {
            throw new IllegalStateException("Invalid protobuf, output domain does not match the converter, found " + carrier.outputDomain().getClass() + " and " + converterWitness);
        }
        List<Booster> models = new ArrayList<>();
        for (ByteString b : proto.getModelsList()) {
            models.add(XGBoost.loadModel(b.toByteArray()));
        }
        if (models.isEmpty()) {
            throw new IllegalStateException("Invalid protobuf, no XGBoost models were found");
        }

        @SuppressWarnings({"rawtypes","unchecked"}) // guarded by getClass check on the converter and domain above
        XGBoostModel<?> model = new XGBoostModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),carrier.outputDomain(),models,converter);
        return model;
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

    /**
     * Creates objects to report feature importance metrics for XGBoost. See the documentation of {@link XGBoostFeatureImportance}
     * for more information on what those metrics mean. Typically this list will contain a single instance for the entire
     * model. For multidimensional regression the list will have one entry per dimension, in dimension order.
     * @return The feature importance object(s).
     */
    public List<XGBoostFeatureImportance> getFeatureImportance() {
        return models.stream().map(b -> new XGBoostFeatureImportance(b, this)).collect(Collectors.toList());
    }

    @Override
    public Map<String, List<Pair<String,Double>>> getTopFeatures(int n) {
        try {
            int maxFeatures = n < 0 ? featureIDMap.size() : n;
            Map<String, List<Pair<String,Double>>> map = new HashMap<>();
            for (int i = 0; i < models.size(); i++) {
                Booster model = models.get(i);
                Map<String, MutableDouble> outputMap = new HashMap<>();
                Map<String, Integer> xgboostMap = model.getFeatureScore("");
                for (Map.Entry<String,Integer> f : xgboostMap.entrySet()) {
                    int id = Integer.parseInt(f.getKey().substring(1));
                    String name = featureIDMap.get(id).getName();
                    MutableDouble curVal = outputMap.computeIfAbsent(name,(k)->new MutableDouble());
                    curVal.increment(f.getValue());
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

                if (models.size() == 1) {
                    map.put(Model.ALL_OUTPUTS, list);
                } else {
                    String dimensionName = outputIDInfo.getOutput(i).toString();
                    map.put(dimensionName, list);
                }
            }

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
            return XGBoost.loadModel(serialisedBooster);
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

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        XGBoostModelProto.Builder modelBuilder = XGBoostModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.setConverter(converter.serialize());
        try {
            for (Booster b : models) {
                modelBuilder.addModels(ByteString.copyFrom(b.toByteArray()));
            }
        } catch (XGBoostError e) {
            throw new IllegalStateException("Failed to serialize XGBoost model");
        }

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(XGBoostModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
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
            this.models = new ArrayList<>();
            int numModels = in.readInt();
            for (int i = 0; i < numModels; i++) {
                // Now read in the byte array and rebuild each Booster
                byte[] serialisedBooster = (byte[]) in.readObject();
                this.models.add(XGBoost.loadModel(serialisedBooster));
            }
            try {
                Class<?> regressionClass = Class.forName("org.tribuo.regression.ImmutableRegressionInfo");
                String tribuoVersion = (String) provenance.getTrainerProvenance().getInstanceValues().get(TrainerProvenance.TRIBUO_VERSION_STRING).getValue();
                if (regressionClass.isInstance(outputIDInfo) && !regression41MappingFix &&
                        (tribuoVersion.startsWith("4.0.0") || tribuoVersion.startsWith("4.0.1") || tribuoVersion.startsWith("4.0.2") || tribuoVersion.startsWith("4.1.0")
                                // This is explicit to catch the test model which has a 4.1.1-SNAPSHOT Tribuo version.
                                || tribuoVersion.equals("4.1.1-SNAPSHOT"))) {
                    // rewrite the model stream
                    regression41MappingFix = true;
                    int[] mapping = (int[]) regressionClass.getDeclaredMethod("getIDtoNaturalOrderMapping").invoke(outputIDInfo);
                    List<Booster> copy = new ArrayList<>(models);
                    for (int i = 0; i < mapping.length; i++) {
                        copy.set(i,models.get(mapping[i]));
                    }
                    this.models = copy;
                }
            } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
                throw new RuntimeException("Failed to rewrite 4.1.0 or earlier regression model due to a reflection failure.",e);
            } catch (ClassNotFoundException e) {
                // pass as this isn't a regression model as otherwise it would have thrown ClassNotFoundException
                // during the reading of the input stream.
            }
        } catch (XGBoostError e) {
            throw new IOException("Failed to deserialize the XGBoost model",e);
        }
    }
}
