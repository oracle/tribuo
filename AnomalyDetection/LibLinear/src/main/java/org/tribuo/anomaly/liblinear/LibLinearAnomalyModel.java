/*
 * Copyright (c) 2021, 2022 Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.anomaly.liblinear;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.Excuse;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.Prediction;
import org.tribuo.anomaly.Event;
import org.tribuo.common.liblinear.LibLinearModel;
import org.tribuo.common.liblinear.LibLinearTrainer;
import org.tribuo.common.liblinear.protos.LibLinearModelProto;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.provenance.ModelProvenance;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.logging.Logger;

/**
 * A {@link Model} which wraps a LibLinear-java anomaly detection model.
 * <p>
 * It disables the LibLinear debug output as it's very chatty.
 * <p>
 * See:
 * <pre>
 * Fan RE, Chang KW, Hsieh CJ, Wang XR, Lin CJ.
 * "LIBLINEAR: A library for Large Linear Classification"
 * Journal of Machine Learning Research, 2008.
 * </pre>
 * and for the original algorithm:
 * <pre>
 * Cortes C, Vapnik V.
 * "Support-Vector Networks"
 * Machine Learning, 1995.
 * </pre>
 */
public class LibLinearAnomalyModel extends LibLinearModel<Event> {
    private static final long serialVersionUID = 3L;

    private static final Logger logger = Logger.getLogger(LibLinearAnomalyModel.class.getName());

    LibLinearAnomalyModel(String name, ModelProvenance description, ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<Event> outputIDInfo, List<de.bwaldvogel.liblinear.Model> models) {
        super(name, description, featureIDMap, outputIDInfo, false, models);
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LibLinearAnomalyModel deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        if (!"org.tribuo.anomaly.liblinear.LibLinearAnomalyModel".equals(className)) {
            throw new IllegalStateException("Invalid protobuf, this class can only deserialize LibLinearAnomalyModel");
        }
        LibLinearModelProto proto = message.unpack(LibLinearModelProto.class);

        ModelDataCarrier<?> carrier = ModelDataCarrier.deserialize(proto.getMetadata());
        if (!carrier.outputDomain().getOutput(0).getClass().equals(Event.class)) {
            throw new IllegalStateException("Invalid protobuf, output domain is not an anomaly domain, found " + carrier.outputDomain().getClass());
        }
        @SuppressWarnings("unchecked") // guarded by getClass
        ImmutableOutputInfo<Event> outputDomain = (ImmutableOutputInfo<Event>) carrier.outputDomain();

        if (proto.getModelsCount() != 1) {
            throw new IllegalStateException("Invalid protobuf, expected 1 model, found " + proto.getModelsCount());
        }
        try {
            ByteArrayInputStream bais = new ByteArrayInputStream(proto.getModels(0).toByteArray());
            ObjectInputStream ois = new ObjectInputStream(bais);
            de.bwaldvogel.liblinear.Model model = (de.bwaldvogel.liblinear.Model) ois.readObject();
            ois.close();
            return new LibLinearAnomalyModel(carrier.name(),carrier.provenance(),carrier.featureDomain(),outputDomain,Collections.singletonList(model));
        } catch (IOException | ClassNotFoundException e) {
            throw new IllegalStateException("Invalid protobuf, failed to deserialize liblinear model", e);
        }
    }

    @Override
    public Prediction<Event> predict(Example<Event> example) {
        FeatureNode[] features = LibLinearTrainer.exampleToNodes(example, featureIDMap, null);
        // Bias feature is always set
        if (features.length == 1) {
            throw new IllegalArgumentException("No features found in Example " + example.toString());
        }

        de.bwaldvogel.liblinear.Model model = models.get(0);
        double[] score = new double[1];
        double prediction = Linear.predictValues(model, features, score);
        if (prediction < 0.0) {
            return new Prediction<>(new Event(Event.EventType.ANOMALOUS,score[0]),features.length,example);
        } else {
            return new Prediction<>(new Event(Event.EventType.EXPECTED,score[0]),features.length,example);
        }
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        int maxFeatures = n < 0 ? featureIDMap.size() : n;
        de.bwaldvogel.liblinear.Model model = models.get(0);
        double[] featureWeights = model.getFeatureWeights();

        Comparator<Pair<String, Double>> comparator = Comparator.comparingDouble(p -> Math.abs(p.getB()));
        
        /*
         * Liblinear stores its weights as follows
         * +------------------+------------------+------------+
         * | nr_class weights | nr_class weights |  ...
         * | for 1st feature  | for 2nd feature  |
         * +------------------+------------------+------------+
         *
         * If bias &gt;= 0, x becomes [x; bias]. The number of features is
         * increased by one, so w is a (nr_feature+1)*nr_class array. The
         * value of bias is stored in the variable bias.
         */

        Map<String, List<Pair<String, Double>>> map = new HashMap<>();
        int numFeatures = model.getNrFeature();
        PriorityQueue<Pair<String, Double>> q = new PriorityQueue<>(maxFeatures, comparator);

        for (int i = 0; i < numFeatures; i++) {
            Pair<String, Double> cur = new Pair<>(featureIDMap.get(i).getName(), featureWeights[i]);
            if (q.size() < maxFeatures) {
                q.offer(cur);
            } else if (comparator.compare(cur, q.peek()) > 0) {
                q.poll();
                q.offer(cur);
            }
        }
        List<Pair<String, Double>> list = new ArrayList<>();
        while (q.size() > 0) {
            list.add(q.poll());
        }
        Collections.reverse(list);
        map.put(Event.EventType.ANOMALOUS.toString(), list);

        List<Pair<String, Double>> otherList = new ArrayList<>();
        for (Pair<String, Double> f : list) {
            Pair<String, Double> otherF = new Pair<>(f.getA(), -f.getB());
            otherList.add(otherF);
        }
        map.put(Event.EventType.EXPECTED.toString(), otherList);
        return map;
    }

    @Override
    protected LibLinearAnomalyModel copy(String newName, ModelProvenance newProvenance) {
        return new LibLinearAnomalyModel(newName,newProvenance,featureIDMap,outputIDInfo,Collections.singletonList(copyModel(models.get(0))));
    }

    @Override
    protected double[][] getFeatureWeights() {
        double[][] featureWeights = new double[1][];
        featureWeights[0] = models.get(0).getFeatureWeights();
        return featureWeights;
    }

    /**
     * The call to model.getFeatureWeights in the public methods copies the
     * weights array so this inner method exists to save the copy in getExcuses.
     * <p>
     * If it becomes a problem then we could cache the feature weights in the
     * model.
     * @param e The example.
     * @param allFeatureWeights The feature weights.
     * @return An excuse for this example.
     */
    @Override
    protected Excuse<Event> innerGetExcuse(Example<Event> e, double[][] allFeatureWeights) {
        de.bwaldvogel.liblinear.Model model = models.get(0);
        double[] featureWeights = allFeatureWeights[0];

        Prediction<Event> prediction = predict(e);
        Map<String, List<Pair<String, Double>>> weightMap = new HashMap<>();

        List<Pair<String, Double>> posScores = new ArrayList<>();
        List<Pair<String, Double>> negScores = new ArrayList<>();
        for (Feature f : e) {
            int id = featureIDMap.getID(f.getName());
            if (id > -1) {
                double score = featureWeights[id] * f.getValue();
                posScores.add(new Pair<>(f.getName(), score));
                negScores.add(new Pair<>(f.getName(), -score));
            }
        }
        posScores.sort((o1, o2) -> o2.getB().compareTo(o1.getB()));
        negScores.sort((o1, o2) -> o2.getB().compareTo(o1.getB()));
        weightMap.put(Event.EventType.ANOMALOUS.toString(),posScores);
        weightMap.put(Event.EventType.EXPECTED.toString(),negScores);

        return new Excuse<>(e, prediction, weightMap);
    }
}
