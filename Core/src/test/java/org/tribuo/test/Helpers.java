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

package org.tribuo.test;

import com.google.protobuf.Message;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.junit.jupiter.api.Assertions;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.Prediction;
import org.tribuo.impl.ListExample;
import org.tribuo.protos.ProtoSerializable;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.DatasetProto;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.protos.core.SequenceDatasetProto;
import org.tribuo.protos.core.SequenceModelProto;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceModel;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Test helpers
 */
public final class Helpers {

    private static final Logger logger = Logger.getLogger(Helpers.class.getName());

    private Helpers() {}

    /**
     * Makes a feature map by observing each feature once with the value 1.0.
     * @param features The feature names.
     * @return An immutable feature map with all the features observed.
     */
    public static ImmutableFeatureMap mkFeatureMap(String... features) {
        MutableFeatureMap fmap = new MutableFeatureMap();
        for (String s : features) {
            fmap.add(s,1.0);
        }
        return new ImmutableFeatureMap(fmap);
    }

    public static Example<MockOutput> mkExample(MockOutput label, String... features) {
        Example<MockOutput> ex = new ListExample<>(label);
        Map<String, Integer> counts = new HashMap<>();
        for (String s : features) {
            counts.put(s, counts.getOrDefault(s, 0)+1);
        }
        for (Map.Entry<String, Integer> kv : counts.entrySet()) {
            ex.add(new Feature(kv.getKey(), 1d*kv.getValue()));
        }
        return ex;
    }

    /**
     * Checks for equality between two sequence datasets.
     * <p>
     * Equality is defined as all examples are equal, in the same order, the output factories are the same and the
     * feature & output domains are equal. Provenance is not compared, nor are other properties of the sequence dataset.
     * @param first The first dataset.
     * @param second The second dataset.
     * @return True if the datasets are equal.
     * @param <T> The output type.
     */
    public static <T extends Output<T>> boolean sequenceDatasetEquals(SequenceDataset<T> first, SequenceDataset<T> second) {
        if (first.size() != second.size()) {
            return false;
        }
        for (int i = 0; i < first.size(); i++) {
            if (!first.getExample(i).equals(second.getExample(i))) {
                return false;
            }
        }
        if (!first.getOutputFactory().equals(second.getOutputFactory())) {
            return false;
        }
        if (!first.getFeatureMap().equals(second.getFeatureMap())) {
            return false;
        }
        return first.getOutputInfo().equals(second.getOutputInfo());
    }

    /**
     * Checks for equality between two datasets.
     * <p>
     * Equality is defined as all examples are equal, in the same order, the output factories are the same and the
     * feature & output domains are equal. Provenance is not compared, nor are other properties of the dataset.
     * @param first The first dataset.
     * @param second The second dataset.
     * @return True if the datasets are equal.
     * @param <T> The output type.
     */
    public static <T extends Output<T>> boolean datasetEquals(Dataset<T> first, Dataset<T> second) {
        if (first.size() != second.size()) {
            return false;
        }
        for (int i = 0; i < first.size(); i++) {
            if (!first.getExample(i).equals(second.getExample(i))) {
                return false;
            }
        }
        if (!first.getOutputFactory().equals(second.getOutputFactory())) {
            return false;
        }
        if (!first.getFeatureMap().equals(second.getFeatureMap())) {
            return false;
        }
        return first.getOutputInfo().equals(second.getOutputInfo());
    }

    /**
     * Takes an object that is both {@link Provenancable} and {@link Configurable} and tests whether the configuration
     * and provenance representations are the same using {@link ConfigurationData#structuralEquals(List, List, String, String)}.
     * @param itm The object whose equality is to be tested
     */
    public static <P extends ConfiguredObjectProvenance, C extends Configurable & Provenancable<P>> void testConfigurableRoundtrip(C itm) {
        ConfigurationManager cm = new ConfigurationManager();
        String name = cm.importConfigurable(itm, "item");
        List<ConfigurationData> configData = cm.getComponentNames().stream()
                .map(cm::getConfigurationData)
                .filter(Optional::isPresent)
                .map(Optional::get)
                .collect(Collectors.toList());

        List<ConfigurationData> provenData = ProvenanceUtil.extractConfiguration(itm.getProvenance());

        assertTrue(ConfigurationData.structuralEquals(configData, provenData, name, provenData.get(0).getName()));
    }

    public static void testProvenanceMarshalling(ObjectProvenance inputProvenance) {
        List<ObjectMarshalledProvenance> provenanceList = ProvenanceUtil.marshalProvenance(inputProvenance);
        ObjectProvenance unmarshalledProvenance = ProvenanceUtil.unmarshalProvenance(provenanceList);
        assertEquals(unmarshalledProvenance,inputProvenance);
    }

    @SuppressWarnings({"unchecked","rawtypes"})
    public static <T extends Output<T>> SequenceDataset<T> testSequenceDatasetSerialization(SequenceDataset<T> dataset) {
        SequenceDatasetProto proto = dataset.serialize();
        SequenceDataset deser = ProtoUtil.deserialize(proto);
        assertEquals(dataset.getClass(),deser.getClass());
        assertFalse(dataset == deser);
        assertTrue(sequenceDatasetEquals(dataset, deser));
        return deser;
    }

    @SuppressWarnings({"unchecked","rawtypes"})
    public static <T extends Output<T>> Dataset<T> testDatasetSerialization(Dataset<T> dataset) {
        DatasetProto proto = dataset.serialize();
        Dataset deser = ProtoUtil.deserialize(proto);
        assertEquals(dataset.getClass(),deser.getClass());
        assertFalse(dataset == deser);
        assertTrue(datasetEquals(dataset, deser));
        return deser;
    }

    public static <U extends Message, T extends ProtoSerializable<U>> T testProtoSerialization(T obj) {
        U proto = obj.serialize();
        T deser = ProtoUtil.deserialize(proto);
        assertEquals(obj,deser);
        return deser;
    }


    public static <T extends Output<T>> Model<T> testModelProtoSerialization(Model<T> model, Class<T> outputClazz, Iterable<Example<T>> data) {
        return testModelProtoSerialization(model, outputClazz, data, 1e-15);
    }

    public static <T extends Output<T>> Model<T> testModelProtoSerialization(Model<T> model, Class<T> outputClazz, Iterable<Example<T>> data, double tolerance) {
        // test provenance marshalling
        testProvenanceMarshalling(model.getProvenance());

        // serialize to proto
        ModelProto proto = model.serialize();

        // deserialize from proto
        Model<?> deserializedModel = Model.deserialize(proto);

        // check provenance is equal
        assertEquals(model.getProvenance(), deserializedModel.getProvenance());
        // validate that the model is still of the right type
        assertTrue(deserializedModel.validate(outputClazz));
        Model<T> deserModel = deserializedModel.castModel(outputClazz);

        // validate the predictions are the same
        List<Prediction<T>> modelPreds = model.predict(data);
        List<Prediction<T>> deserPreds = deserModel.predict(data);
        assertEquals(modelPreds.size(),deserPreds.size());
        for (int i = 0; i < modelPreds.size(); i++) {
            Prediction<T> cur = modelPreds.get(i);
            Prediction<T> other = deserPreds.get(i);
            assertTrue(cur.distributionEquals(other, tolerance));
        }

        return deserModel;
    }

    public static <T extends Output<T>> SequenceModel<T> testSequenceModelProtoSerialization(SequenceModel<T> model, Class<T> outputClazz, SequenceDataset<T> data) {
        // test provenance marshalling
        testProvenanceMarshalling(model.getProvenance());

        // serialize to proto
        SequenceModelProto proto = model.serialize();

        // deserialize from proto
        SequenceModel<?> deserializedModel = SequenceModel.deserialize(proto);

        // check provenance is equal
        assertEquals(model.getProvenance(), deserializedModel.getProvenance());
        // validate that the model is still of the right type
        assertTrue(deserializedModel.validate(outputClazz));
        SequenceModel<T> deserModel = deserializedModel.castModel(outputClazz);

        // validate the predictions are the same
        List<List<Prediction<T>>> modelPreds = model.predict(data);
        List<List<Prediction<T>>> deserPreds = deserModel.predict(data);
        assertEquals(modelPreds.size(),deserPreds.size());
        for (int i = 0; i < modelPreds.size(); i++) {
            List<Prediction<T>> innerModelPreds = modelPreds.get(i);
            List<Prediction<T>> innerDeserPreds = deserPreds.get(i);
            assertEquals(innerModelPreds.size(), innerDeserPreds.size());
            for (int j = 0; j < innerModelPreds.size(); j++) {
                Prediction<T> cur = innerModelPreds.get(j);
                Prediction<T> other = innerDeserPreds.get(j);
                assertTrue(cur.distributionEquals(other));
            }
        }

        return deserModel;
    }

    public static <T extends Output<T>> void testModelSerialization(Model<T> model, Class<T> outputClazz) {
        // test provenance marshalling
        testProvenanceMarshalling(model.getProvenance());

        // write to byte array
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(baos))) {
            oos.writeObject(model);
        } catch (IOException ex) {
            logger.severe("IOException when writing out model");
            Assertions.fail("Failed to serialize model class " + model.getClass().toString(), ex);
        }

        // Extract the byte array
        byte[] modelSer = baos.toByteArray();

        // read model from byte array
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new ByteArrayInputStream(modelSer)))) {
            Model<?> deserializedModel = (Model<?>) ois.readObject();
            // check provenance is equal
            assertEquals(model.getProvenance(), deserializedModel.getProvenance());
            // validate that the model is still of the right type
            assertTrue(deserializedModel.validate(outputClazz));
            if (deserializedModel instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) deserializedModel).close();
                } catch (Exception ex) {
                    logger.severe("Exception thrown when closing model");
                    Assertions.fail("Failed to close deserialized model " + model.getClass().toString(),ex);
                }
            }
        } catch (IOException ex) {
            logger.severe("IOException when reading in model");
            Assertions.fail("Failed to deserialize model class " + model.getClass().toString(), ex);
        } catch (ClassNotFoundException ex) {
            logger.severe("ClassNotFoundException when reading in model");
            Assertions.fail("Failed to deserialize model class " + model.getClass().toString(), ex);
        }
    }

    public static <T extends Output<T>> void testSequenceModelSerialization(SequenceModel<T> model, Class<T> outputClazz) {
        // write to byte array
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(baos))) {
            oos.writeObject(model);
        } catch (IOException ex) {
            logger.severe("IOException when writing out model");
            Assertions.fail("Failed to serialize sequence model class " + model.getClass().toString(), ex);
        }

        // Extract the byte array
        byte[] modelSer = baos.toByteArray();

        // read model from byte array
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new ByteArrayInputStream(modelSer)))) {
            SequenceModel<?> deserializedModel = (SequenceModel<?>) ois.readObject();
            // check provenance is equal
            assertEquals(model.getProvenance(), deserializedModel.getProvenance());
            // validate that the model is still of the right type
            assertTrue(deserializedModel.validate(outputClazz));
            if (deserializedModel instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) deserializedModel).close();
                } catch (Exception ex) {
                    logger.severe("Exception thrown when closing model");
                    Assertions.fail("Failed to close deserialized model " + model.getClass().toString(),ex);
                }
            }
        } catch (IOException ex) {
            logger.severe("IOException when reading in model");
            Assertions.fail("Failed to deserialize sequence model class " + model.getClass().toString(), ex);
        } catch (ClassNotFoundException ex) {
            logger.severe("ClassNotFoundException when reading in model");
            Assertions.fail("Failed to deserialize sequence model class " + model.getClass().toString(), ex);
        }
    }

    /**
     * Compares two top feature lists according to the specified tolerances returning true when the lists have the
     * same elements and the difference between the scores is within the tolerance.
     * <p>
     * Mostly used when refactoring implementations to compare the new and old one.
     * @param first The first feature list.
     * @param second The second feature list.
     * @param tolerance The tolerance for the scores.
     * @return True if the feature lists are equal.
     */
    public static boolean topFeaturesEqual(Map<String, List<Pair<String,Double>>> first, Map<String, List<Pair<String,Double>>> second, double tolerance)  {
        if (first.size() == second.size() && first.keySet().containsAll(second.keySet())) {
            // keys the same, now check lists
            for (Map.Entry<String, List<Pair<String, Double>>> e : first.entrySet()) {
                List<Pair<String,Double>> firstList = e.getValue();
                List<Pair<String,Double>> secondList = second.get(e.getKey());
                if (firstList.size() == secondList.size()) {
                    // Now compare lists
                    for (int i = 0; i < firstList.size(); i++) {
                        Pair<String, Double> firstPair = firstList.get(i);
                        Pair<String, Double> secondPair = secondList.get(i);
                        if (firstPair.getA().equals(secondPair.getA())) {
                            double diff = Math.abs(firstPair.getB() - secondPair.getB());
                            if (diff > tolerance) {
                                return false;
                            }
                        } else {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }
}
