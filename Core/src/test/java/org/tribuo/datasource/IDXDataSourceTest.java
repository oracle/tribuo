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

package org.tribuo.datasource;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import org.junit.jupiter.api.Test;
import org.tribuo.FeatureMap;
import org.tribuo.MutableDataset;
import org.tribuo.OutputInfo;
import org.tribuo.datasource.IDXDataSource.IDXData;
import org.tribuo.datasource.IDXDataSource.IDXType;
import org.tribuo.test.MockOutput;
import org.tribuo.test.MockOutputFactory;

import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.fail;

/**
 *
 */
public class IDXDataSourceTest {

    private static final MockOutputFactory factory = new MockOutputFactory();

    public static void generateOutputData() throws IOException {
        IDXData data = IDXData.createIDXData(IDXType.BYTE, new int[]{4}, new double[]{0, 1, 1, 0});
        data.save(Paths.get("./outputs.idx"), false);
        data = IDXData.createIDXData(IDXType.BYTE, new int[]{8}, new double[]{0, 1, 1, 0, 2, 1, 1, 2});
        data.save(Paths.get("./outputs-long.idx"), false);
    }

    public static void generateByteData() throws IOException {
        IDXData data = IDXData.createIDXData(IDXType.BYTE, new int[]{4, 4}, new double[]{1, 2, 3, 4, 4, 3, 2, 1, 5, 6, 7, 8, 8, 7, 6, 5});
        data.save(Paths.get("./byte.idx"), false);
        data = IDXData.createIDXData(IDXType.BYTE, new int[]{4, 2, 2}, new double[]{1, 2, 3, 4, 4, 3, 2, 1, 5, 6, 7, 8, 8, 7, 6, 5});
        data.save(Paths.get("./byte-mat.idx"), false);
    }

    public static void generateIntData() throws IOException {
        IDXData data = IDXData.createIDXData(IDXType.INT, new int[]{4, 4}, new double[]{1234, 2, 3, 4, 4321, 3, 2, 1, 5, 6789, 7, 8, 8765, 7, 6, 5});
        data.save(Paths.get("./int.idx"), false);
    }

    public static void generateInvalidIDX() throws IOException {
        IDXData data = new IDXData(IDXType.INT, new int[]{5, 2}, new double[]{1, 2, 3, 4});
        data.save(Paths.get("./too-little-data.idx"), false);
        data = new IDXData(IDXType.INT, new int[]{5}, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9});
        data.save(Paths.get("./too-much-data.idx"), false);
    }

    public static void generateNonsenseIDX() throws IOException {
        // First one has an invalid magic byte
        try (DataOutputStream stream = new DataOutputStream(new FileOutputStream(Paths.get("invalid-magic-byte.idx").toFile()))) {
            stream.writeShort(5);
            stream.writeByte(IDXType.INT.value);
            stream.writeByte(2);
            stream.writeInt(2);
            stream.writeInt(2);
            stream.writeInt(1);
            stream.writeInt(2);
            stream.writeInt(3);
            stream.writeInt(4);
        }

        // Second has an invalid type byte
        try (DataOutputStream stream = new DataOutputStream(new FileOutputStream(Paths.get("invalid-type-byte.idx").toFile()))) {
            stream.writeShort(0);
            stream.writeByte(128);
            stream.writeByte(2);
            stream.writeInt(2);
            stream.writeInt(2);
            stream.writeInt(1);
            stream.writeInt(2);
            stream.writeInt(3);
            stream.writeInt(4);
        }

        // Third has an invalid dimension byte
        try (DataOutputStream stream = new DataOutputStream(new FileOutputStream(Paths.get("invalid-dim-byte.idx").toFile()))) {
            stream.writeShort(0);
            stream.writeByte(IDXType.INT.value);
            stream.writeByte(-2);
            stream.writeInt(2);
            stream.writeInt(2);
            stream.writeInt(1);
            stream.writeInt(2);
            stream.writeInt(3);
            stream.writeInt(4);
        }

        // Fourth has a different number of dimensions than expected
        try (DataOutputStream stream = new DataOutputStream(new FileOutputStream(Paths.get("incorrect-num-dims.idx").toFile()))) {
            stream.writeShort(0);
            stream.writeByte(IDXType.INT.value);
            stream.writeByte(3);
            stream.writeInt(2);
            stream.writeInt(2);
            stream.writeInt(1);
            stream.writeInt(2);
            stream.writeInt(3);
            stream.writeInt(4);
        }

        // Fifth has no data
        try (DataOutputStream stream = new DataOutputStream(new FileOutputStream(Paths.get("no-data.idx").toFile()))) {
            stream.writeShort(0);
            stream.writeByte(IDXType.INT.value);
            stream.writeByte(2);
            stream.writeInt(2);
            stream.writeInt(2);
        }
    }

    // Writes out the test data files
    public static void main(String[] args) throws IOException {
        generateByteData();
        generateIntData();
        generateOutputData();
        generateInvalidIDX();
        generateNonsenseIDX();
    }

    @Test
    public void testByteLoading() throws IOException, URISyntaxException {
        Path dataFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/byte.idx").toURI());
        Path outputFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/outputs.idx").toURI());
        testIDXLoading(dataFile, outputFile);
        dataFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/byte-mat.idx").toURI());
        testIDXLoading(dataFile, outputFile);
    }

    @Test
    public void testIntLoading() throws IOException, URISyntaxException {
        Path dataFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/int.idx").toURI());
        Path outputFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/outputs.idx").toURI());
        testIDXLoading(dataFile, outputFile);
    }

    private void testIDXLoading(Path dataFile, Path outputFile) throws IOException {
        IDXDataSource<MockOutput> source = new IDXDataSource<>(dataFile, outputFile, factory);

        MutableDataset<MockOutput> dataset = new MutableDataset<>(source);

        FeatureMap fmap = dataset.getFeatureMap();
        assertEquals(4, fmap.size());

        OutputInfo<MockOutput> info = dataset.getOutputInfo();
        assertEquals(2, info.size());
    }

    @Test
    public void testInvalidCombination() throws IOException, URISyntaxException {
        Path dataFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/int.idx").toURI());
        Path outputFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/outputs-long.idx").toURI());
        assertThrows(IllegalStateException.class, () -> new IDXDataSource<>(dataFile, outputFile, factory));
    }

    @Test
    public void testInvalidIDX() throws URISyntaxException {
        Path dataFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/too-much-data.idx").toURI());
        assertThrows(IllegalStateException.class, () -> IDXDataSource.readData(dataFile));
        Path otherDataFile = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/too-little-data.idx").toURI());
        assertThrows(IllegalStateException.class, () -> IDXDataSource.readData(otherDataFile));
    }

    @Test
    public void testNonsenseIDX() throws URISyntaxException {
        // First one has an invalid magic byte
        Path first = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/invalid-magic-byte.idx").toURI());
        assertThrows(IllegalStateException.class, () -> IDXDataSource.readData(first));

        // Second has an invalid type byte
        Path second = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/invalid-type-byte.idx").toURI());
        assertThrows(IllegalArgumentException.class, () -> IDXDataSource.readData(second));

        // Third has an invalid dimension byte
        Path third = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/invalid-dim-byte.idx").toURI());
        assertThrows(IllegalStateException.class, () -> IDXDataSource.readData(third));

        // Fourth has a different number of dimensions than expected
        // This is tricky to test for as there is no sentinel between the dimensions and the data
        // so it will either exhibit as EOF if the dimensions read off the end of the file, or
        // IllegalStateException if there is data left at the end of the file.
        Path fourth = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/incorrect-num-dims.idx").toURI());
        assertThrows(IllegalStateException.class, () -> IDXDataSource.readData(fourth));

        // Fifth has no data
        Path fifth = Paths.get(IDXDataSourceTest.class.getResource("/org/tribuo/datasource/no-data.idx").toURI());
        assertThrows(IllegalStateException.class, () -> IDXDataSource.readData(fifth));
    }

    @Test
    public void testFileNotFound() throws IOException {
        // First check that the configuration system throws out of postConfig
        ConfigurationManager cm = new ConfigurationManager("/org/tribuo/datasource/config.xml");

        try {
            @SuppressWarnings("unchecked") // this config file is in the tests, we know the type
            IDXDataSource<MockOutput> tmp = (IDXDataSource<MockOutput>) cm.lookup("train");
            fail("Should have thrown PropertyException");
        } catch (PropertyException e) {
            if (!e.getMessage().contains("Failed to load from path - ")) {
                fail("Incorrect exception message",e);
            }
        } catch (RuntimeException e) {
            fail("Incorrect exception thrown",e);
        }

        // Next check the constructor throws
        MockOutputFactory factory = new MockOutputFactory();
        try {
            IDXDataSource<MockOutput> tmp = new IDXDataSource<>(Paths.get("these-features-dont-exist"), Paths.get("these-outputs-dont-exist"), factory);
            fail("Should have thrown FileNotFoundException");
        } catch (FileNotFoundException e) {
            if (!e.getMessage().contains("Failed to load from path - ")) {
                fail("Incorrect exception message",e);
            }
        }
    }

}
