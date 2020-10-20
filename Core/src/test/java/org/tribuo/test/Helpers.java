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

package org.tribuo.test;

import org.junit.jupiter.api.Assertions;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.impl.ListExample;
import org.tribuo.sequence.SequenceModel;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Test helpers
 */
public final class Helpers {

    private static final Logger logger = Logger.getLogger(Helpers.class.getName());

    private Helpers() {}

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

    public static <T extends Output<T>> void testModelSerialization(Model<T> model, Class<T> outputClazz) {
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
        } catch (IOException ex) {
            logger.severe("IOException when reading in model");
            Assertions.fail("Failed to deserialize sequence model class " + model.getClass().toString(), ex);
        } catch (ClassNotFoundException ex) {
            logger.severe("ClassNotFoundException when reading in model");
            Assertions.fail("Failed to deserialize sequence model class " + model.getClass().toString(), ex);
        }
    }
}
