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

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.EnumProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.IOUtil;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataSourceProvenance;

import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Path;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.logging.Logger;
import java.util.zip.GZIPOutputStream;

/**
 * A DataSource which can read IDX formatted data (i.e., MNIST).
 * <p>
 * Transparently reads GZipped files.
 * <p>
 * The file format is defined <a href="http://yann.lecun.com/exdb/mnist/">here</a>.
 */
public final class IDXDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {
    private static final Logger logger = Logger.getLogger(IDXDataSource.class.getName());

    /**
     * The possible IDX input formats.
     */
    public enum IDXType {
        /**
         * An unsigned byte.
         */
        UBYTE((byte) 0x08),
        /**
         * A signed byte.
         */
        BYTE((byte) 0x09),
        /**
         * A 16-bit integer.
         */
        SHORT((byte) 0x0B),
        /**
         * A 32-bit integer.
         */
        INT((byte) 0x0C),
        /**
         * A 32-bit float.
         */
        FLOAT((byte) 0x0D),
        /**
         * A 64-bit float.
         */
        DOUBLE((byte) 0x0E);

        /**
         * The encoded byte value.
         */
        public final byte value;

        IDXType(byte value) {
            this.value = value;
        }

        /**
         * Converts the byte into the enum. Throws IllegalArgumentException if it's
         * not a valid byte.
         *
         * @param input The byte to convert.
         * @return The corresponding enum instance.
         */
        public static IDXType convert(byte input) {
            for (IDXType f : values()) {
                if (f.value == input) {
                    return f;
                }
            }
            throw new IllegalArgumentException("Invalid byte found - " + input);
        }
    }

    @Config(mandatory = true, description = "Path to load the features from.")
    private Path featuresPath;

    @Config(mandatory = true, description = "Path to load the features from.")
    private Path outputPath;

    @Config(mandatory = true, description = "The output factory to use.")
    private OutputFactory<T> outputFactory;

    private final ArrayList<Example<T>> data = new ArrayList<>();

    private IDXType dataType;

    private IDXDataSourceProvenance provenance;

    /**
     * For olcut.
     */
    private IDXDataSource() {}

    /**
     * Constructs an IDXDataSource from the supplied paths.
     *
     * @param featuresPath  The path to the features file.
     * @param outputPath    The path to the output file.
     * @param outputFactory The output factory.
     * @throws IOException If either file cannot be read.
     */
    public IDXDataSource(Path featuresPath, Path outputPath, OutputFactory<T> outputFactory) throws IOException {
        this.outputFactory = outputFactory;
        this.featuresPath = featuresPath;
        this.outputPath = outputPath;
        read();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        read();
    }

    @Override
    public String toString() {
        return "IDXDataSource(featuresPath=" + featuresPath.toString() + ",outputPath=" + outputPath.toString() + ",featureType=" + dataType + ")";
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    public synchronized DataSourceProvenance getProvenance() {
        if (provenance == null) {
            provenance = cacheProvenance();
        }
        return provenance;
    }

    private IDXDataSourceProvenance cacheProvenance() {
        return new IDXDataSourceProvenance(this);
    }

    /**
     * Loads the data.
     *
     * @throws IOException If the files could not be read.
     */
    private void read() throws IOException {
        IDXData features = readData(featuresPath);
        IDXData outputs = readData(outputPath);

        dataType = features.dataType;

        if (features.shape[0] != outputs.shape[0]) {
            throw new IllegalStateException("Features and outputs have different numbers of examples, feature shape = " + Arrays.toString(features.shape) + ", output shape = " + Arrays.toString(outputs.shape));
        }

        // Calculate the example size
        int numFeatures = 1;
        for (int i = 1; i < features.shape.length; i++) {
            numFeatures *= features.shape[i];
        }
        int numOutputs = 1;
        for (int i = 1; i < outputs.shape.length; i++) {
            numOutputs *= outputs.shape[i];
        }

        String[] featureNames = new String[numFeatures];
        int width = ("" + numFeatures).length();
        String formatString = "%0" + width + "d";
        for (int i = 0; i < numFeatures; i++) {
            featureNames[i] = String.format(formatString, i);
        }

        ArrayList<Feature> buffer = new ArrayList<>();
        int featureCounter = 0;
        int outputCounter = 0;
        StringBuilder outputBuilder = new StringBuilder();
        for (int i = 0; i < features.data.length; i++) {
            double curValue = features.data[i];
            if (curValue != 0.0) {
                // Tribuo is sparse, so only create non-zero features
                buffer.add(new Feature(featureNames[featureCounter], curValue));
            }
            featureCounter++;
            if (featureCounter == numFeatures) {
                // fabricate output. Multidimensional outputs expect a comma separated string.
                outputBuilder.setLength(0);
                for (int j = 0; j < numOutputs; j++) {
                    if (j != 0) {
                        outputBuilder.append(',');
                    }
                    // If necessary cast to int to ensure we get a integer out for use as a class label
                    // No-one wants to have MNIST digits with labels "0.0", "1.0" etc.
                    switch (outputs.dataType) {
                        case BYTE:
                        case UBYTE:
                        case SHORT:
                        case INT:
                            outputBuilder.append((int) outputs.data[j + outputCounter]);
                            break;
                        case FLOAT:
                        case DOUBLE:
                            outputBuilder.append(outputs.data[j + outputCounter]);
                            break;
                    }
                }
                outputCounter += numOutputs;
                T output = outputFactory.generateOutput(outputBuilder.toString());

                // create example
                Example<T> example = new ArrayExample<T>(output);
                example.addAll(buffer);
                data.add(example);

                // Clean up
                buffer.clear();
                featureCounter = 0;
            }
        }

        if (featureCounter != 0) {
            throw new IllegalStateException("Failed to process all the features, missing " + (numFeatures - featureCounter) + " values");
        }
    }

    /**
     * Reads a single IDX format file.
     *
     * @param path The path to read.
     * @return The IDXData from the file.
     * @throws IOException If the file could not be read.
     */
    static IDXData readData(Path path) throws IOException {
        InputStream inputStream = IOUtil.getInputStreamForLocation(path.toString());
        if (inputStream == null) {
            throw new FileNotFoundException("Failed to load from path - " + path);
        }
        // DataInputStream.close implicitly closes the InputStream
        try (DataInputStream stream = new DataInputStream(inputStream)) {
            short magicNumber = stream.readShort();
            if (magicNumber != 0) {
                throw new IllegalStateException("Invalid IDX file, magic number was not zero. Found " + magicNumber);
            }
            final byte dataTypeByte = stream.readByte();
            final IDXType dataType = IDXType.convert(dataTypeByte);
            final byte numDimensions = stream.readByte();
            if (numDimensions < 1) {
                throw new IllegalStateException("Invalid number of dimensions, found " + numDimensions);
            }
            final int[] shape = new int[numDimensions];
            int size = 1;
            for (int i = 0; i < numDimensions; i++) {
                shape[i] = stream.readInt();
                if (shape[i] < 1) {
                    throw new IllegalStateException("Invalid shape, found " + Arrays.toString(shape));
                }
                size *= shape[i];
            }
            double[] data = new double[size];
            try {
                for (int i = 0; i < size; i++) {
                    switch (dataType) {
                        case BYTE:
                            data[i] = stream.readByte();
                            break;
                        case UBYTE:
                            data[i] = stream.readUnsignedByte();
                            break;
                        case SHORT:
                            data[i] = stream.readShort();
                            break;
                        case INT:
                            data[i] = stream.readInt();
                            break;
                        case FLOAT:
                            data[i] = stream.readFloat();
                            break;
                        case DOUBLE:
                            data[i] = stream.readDouble();
                            break;
                    }
                }
            } catch (EOFException e) {
                throw new IllegalStateException("Too little data in the file, expected to find " + size + " elements");
            }
            try {
                byte unexpectedByte = stream.readByte();
                throw new IllegalStateException("Too much data in the file");
            } catch (EOFException e) {
                //pass as the stream is exhausted
            }
            return new IDXData(dataType, shape, data);
        }
    }

    /**
     * The number of examples loaded.
     *
     * @return The number of examples.
     */
    public int size() {
        return data.size();
    }

    /**
     * The type of the features that were loaded in.
     *
     * @return The feature type.
     */
    public IDXType getDataType() {
        return dataType;
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return data.iterator();
    }

    /**
     * Java side representation for an IDX file.
     */
    public static class IDXData {
        final IDXType dataType;
        final int[] shape;
        final double[] data;

        /**
         * Constructor, does not validate or copy inputs.
         * Use the factory method.
         * @param dataType The data type.
         * @param shape    The tensor shape.
         * @param data     The data to write.
         */
        IDXData(IDXType dataType, int[] shape, double[] data) {
            this.dataType = dataType;
            this.shape = shape;
            this.data = data;
        }

        /**
         * Constructs an IDXData, validating the input and defensively copying it.
         *
         * @param dataType The data type.
         * @param shape    The tensor shape.
         * @param data     The data to write.
         * @return An IDXData.
         */
        public static IDXData createIDXData(IDXType dataType, int[] shape, double[] data) {
            int[] shapeCopy = Arrays.copyOf(shape, shape.length);
            double[] dataCopy = Arrays.copyOf(data, data.length);
            if (shape.length > 128) {
                throw new IllegalArgumentException("Must have fewer than 128 dimensions");
            }
            int numElements = 1;
            for (int i = 0; i < shapeCopy.length; i++) {
                numElements *= shapeCopy[i];
                if (shapeCopy[i] < 1) {
                    throw new IllegalArgumentException("Invalid shape, all elements must be positive, found " + Arrays.toString(shapeCopy));
                }
            }
            if (numElements != dataCopy.length) {
                throw new IllegalArgumentException("Incorrect number of elements, expected " + numElements + ", found " + dataCopy.length);
            }

            if (dataType != IDXType.DOUBLE) {
                for (int i = 0; i < dataCopy.length; i++) {
                    switch (dataType) {
                        case UBYTE:
                            int tmpU = 0xFF & (int) dataCopy[i];
                            if (dataCopy[i] != tmpU) {
                                throw new IllegalArgumentException("Invalid value at idx " + i + ", could not be converted to unsigned byte");
                            }
                            break;
                        case BYTE:
                            byte tmpB = (byte) dataCopy[i];
                            if (dataCopy[i] != tmpB) {
                                throw new IllegalArgumentException("Invalid value at idx " + i + ", could not be converted to byte");
                            }
                            break;
                        case SHORT:
                            short tmpS = (short) dataCopy[i];
                            if (dataCopy[i] != tmpS) {
                                throw new IllegalArgumentException("Invalid value at idx " + i + ", could not be converted to short");
                            }
                            break;
                        case INT:
                            int tmpI = (int) dataCopy[i];
                            if (dataCopy[i] != tmpI) {
                                throw new IllegalArgumentException("Invalid value at idx " + i + ", could not be converted to int");
                            }
                            break;
                        case FLOAT:
                            float tmpF = (float) dataCopy[i];
                            if (dataCopy[i] != tmpF) {
                                throw new IllegalArgumentException("Invalid value at idx " + i + ", could not be converted to float");
                            }
                            break;
                    }
                }
            }

            return new IDXData(dataType, shape, data);
        }

        /**
         * Writes out this IDXData to the specified path.
         *
         * @param outputPath The path to write to.
         * @param gzip       If true, gzip the output.
         * @throws IOException If the write failed.
         */
        public void save(Path outputPath, boolean gzip) throws IOException {
            try (DataOutputStream ds = makeStream(outputPath, gzip)) {
                // Magic number
                ds.writeShort(0);
                // Data type
                ds.writeByte(dataType.value);
                // Num dimensions
                ds.writeByte(shape.length);

                for (int i = 0; i < shape.length; i++) {
                    ds.writeInt(shape[i]);
                }

                for (int i = 0; i < data.length; i++) {
                    switch (dataType) {
                        case UBYTE:
                            ds.writeByte(0xFF & (int) data[i]);
                            break;
                        case BYTE:
                            ds.writeByte((byte) data[i]);
                            break;
                        case SHORT:
                            ds.writeShort((short) data[i]);
                            break;
                        case INT:
                            ds.writeInt((int) data[i]);
                            break;
                        case FLOAT:
                            ds.writeFloat((float) data[i]);
                            break;
                        case DOUBLE:
                            ds.writeDouble(data[i]);
                            break;
                    }
                }
            }
        }

        private static DataOutputStream makeStream(Path outputPath, boolean gzip) throws IOException {
            OutputStream stream;
            if (gzip) {
                stream = new GZIPOutputStream(new FileOutputStream(outputPath.toFile()));
            } else {
                stream = new FileOutputStream(outputPath.toFile());
            }
            return new DataOutputStream(new BufferedOutputStream(stream));
        }
    }

    /**
     * Provenance class for {@link IDXDataSource}.
     */
    public static final class IDXDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        /**
         * The name of the output file modified time provenance field.
         */
        public static final String OUTPUT_FILE_MODIFIED_TIME = "output-file-modified-time";
        /**
         * The name of the features file modified time provenance field.
         */
        public static final String FEATURES_FILE_MODIFIED_TIME = "features-file-modified-time";
        /**
         * The name of the provenance field for the feature file hash.
         */
        public static final String FEATURES_RESOURCE_HASH = "features-resource-hash";
        /**
         * The name of the provenance field for the output file hash.
         */
        public static final String OUTPUT_RESOURCE_HASH = "output-resource-hash";
        /**
         * The name of the provenance field for the idx feature type.
         */
        public static final String FEATURE_TYPE = "idx-feature-type";

        private final DateTimeProvenance featuresFileModifiedTime;
        private final DateTimeProvenance outputFileModifiedTime;
        private final DateTimeProvenance dataSourceCreationTime;
        private final HashProvenance featuresSHA256Hash;
        private final HashProvenance outputSHA256Hash;
        private final EnumProvenance<IDXType> featureType;

        <T extends Output<T>> IDXDataSourceProvenance(IDXDataSource<T> host) {
            super(host, "DataSource");
            this.outputFileModifiedTime = new DateTimeProvenance(OUTPUT_FILE_MODIFIED_TIME, OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.outputPath.toFile().lastModified()), ZoneId.systemDefault()));
            this.featuresFileModifiedTime = new DateTimeProvenance(FEATURES_FILE_MODIFIED_TIME, OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.featuresPath.toFile().lastModified()), ZoneId.systemDefault()));
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME, OffsetDateTime.now());
            this.featuresSHA256Hash = new HashProvenance(DEFAULT_HASH_TYPE, FEATURES_RESOURCE_HASH, ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE, host.featuresPath));
            this.outputSHA256Hash = new HashProvenance(DEFAULT_HASH_TYPE, OUTPUT_RESOURCE_HASH, ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE, host.outputPath));
            this.featureType = new EnumProvenance<>(FEATURE_TYPE, host.dataType);
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public IDXDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        // Suppressed due to enum provenance cast
        @SuppressWarnings("unchecked")
        private IDXDataSourceProvenance(ExtractedInfo info) {
            super(info);
            this.featuresFileModifiedTime = (DateTimeProvenance) info.instanceValues.get(FEATURES_FILE_MODIFIED_TIME);
            this.outputFileModifiedTime = (DateTimeProvenance) info.instanceValues.get(OUTPUT_FILE_MODIFIED_TIME);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
            this.featuresSHA256Hash = (HashProvenance) info.instanceValues.get(FEATURES_RESOURCE_HASH);
            this.outputSHA256Hash = (HashProvenance) info.instanceValues.get(OUTPUT_RESOURCE_HASH);
            this.featureType = (EnumProvenance<IDXType>) info.instanceValues.get(FEATURE_TYPE);
        }

        /**
         * Separates out the configured and non-configured provenance values.
         * @param map The provenances to separate.
         * @return The extracted provenance information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String, Provenance> map) {
            Map<String, Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters, CLASS_NAME, StringProvenance.class, IDXDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters, HOST_SHORT_NAME, StringProvenance.class, IDXDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String, PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(FEATURES_FILE_MODIFIED_TIME, ObjectProvenance.checkAndExtractProvenance(configuredParameters, FEATURES_FILE_MODIFIED_TIME, DateTimeProvenance.class, IDXDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(OUTPUT_FILE_MODIFIED_TIME, ObjectProvenance.checkAndExtractProvenance(configuredParameters, OUTPUT_FILE_MODIFIED_TIME, DateTimeProvenance.class, IDXDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(DATASOURCE_CREATION_TIME, ObjectProvenance.checkAndExtractProvenance(configuredParameters, DATASOURCE_CREATION_TIME, DateTimeProvenance.class, IDXDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(FEATURES_RESOURCE_HASH, ObjectProvenance.checkAndExtractProvenance(configuredParameters, FEATURES_RESOURCE_HASH, HashProvenance.class, IDXDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(OUTPUT_RESOURCE_HASH, ObjectProvenance.checkAndExtractProvenance(configuredParameters, OUTPUT_RESOURCE_HASH, HashProvenance.class, IDXDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(FEATURE_TYPE, ObjectProvenance.checkAndExtractProvenance(configuredParameters, FEATURE_TYPE, EnumProvenance.class, IDXDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className, hostTypeStringName, configuredParameters, instanceParameters);
        }

        @Override
        public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
            Map<String, PrimitiveProvenance<?>> map = super.getInstanceValues();

            map.put(featuresFileModifiedTime.getKey(), featuresFileModifiedTime);
            map.put(outputFileModifiedTime.getKey(), outputFileModifiedTime);
            map.put(dataSourceCreationTime.getKey(), dataSourceCreationTime);
            map.put(featuresSHA256Hash.getKey(), featuresSHA256Hash);
            map.put(outputSHA256Hash.getKey(), outputSHA256Hash);
            map.put(featureType.getKey(), featureType);

            return map;
        }
    }
}
