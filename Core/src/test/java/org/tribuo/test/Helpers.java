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

package org.tribuo.test;

import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.io.ObjectMarshalledProvenance;
import org.junit.jupiter.api.Assertions;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.MutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.impl.ListExample;
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

        Assertions.assertTrue(ConfigurationData.structuralEquals(configData, provenData, name, provenData.get(0).getName()));
    }


    public static void testProvenanceMarshalling(ObjectProvenance inputProvenance) {
        List<ObjectMarshalledProvenance> provenanceList = ProvenanceUtil.marshalProvenance(inputProvenance);
        ObjectProvenance unmarshalledProvenance = ProvenanceUtil.unmarshalProvenance(provenanceList);
        Assertions.assertEquals(unmarshalledProvenance,inputProvenance);
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
            Assertions.assertEquals(model.getProvenance(), deserializedModel.getProvenance());
            // validate that the model is still of the right type
            Assertions.assertTrue(deserializedModel.validate(outputClazz));
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
            Assertions.assertEquals(model.getProvenance(), deserializedModel.getProvenance());
            // validate that the model is still of the right type
            Assertions.assertTrue(deserializedModel.validate(outputClazz));
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
}
