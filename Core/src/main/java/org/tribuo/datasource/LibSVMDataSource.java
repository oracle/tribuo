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

package org.tribuo.datasource;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataSourceProvenance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * A DataSource which can read LibSVM formatted data.
 * <p>
 * It also provides a static save method which writes LibSVM format data.
 * <p>
 * This class can read libsvm files which are zero-indexed or one-indexed, and the
 * parsed result is available after construction. When loading testing data it's
 * best to use the maxFeatureID from the training data (or the number of features
 * in the model) to ensure that the feature names are formatted with the appropriate
 * number of leading zeros.
 */
public final class LibSVMDataSource<T extends Output<T>> implements ConfigurableDataSource<T> {
    private static final Logger logger = Logger.getLogger(LibSVMDataSource.class.getName());

    private static final Pattern splitPattern = Pattern.compile("\\s+");

    // url is the store of record.
    @Config(description="URL to load the data from. Either this or path must be set.")
    private URL url;

    @Config(description="Path to load the data from. Either this or url must be set.")
    private Path path;

    @Config(mandatory = true, description="The output factory to use.")
    private OutputFactory<T> outputFactory;

    @Config(description="Set to true if the features are zero indexed.")
    private boolean zeroIndexed;

    @Config(description="Sets the maximum feature id to load from the file.")
    private int maxFeatureID = Integer.MIN_VALUE;

    private boolean rangeSet;
    private int minFeatureID = Integer.MAX_VALUE;

    private final ArrayList<Example<T>> data = new ArrayList<>();

    private LibSVMDataSourceProvenance provenance;

    /**
     * For olcut.
     */
    private LibSVMDataSource() {}

    /**
     * Constructs a LibSVMDataSource from the supplied path and output factory.
     * @param path The path to load.
     * @param outputFactory The output factory to use.
     * @throws IOException If the file could not be read or is an invalid format.
     */
    public LibSVMDataSource(Path path, OutputFactory<T> outputFactory) throws IOException {
        this(path.normalize(),path.normalize().toUri().toURL(),outputFactory,false,false,0);
    }

    /**
     * Constructs a LibSVMDataSource from the supplied path and output factory.
     * <p>
     * Also allows control over the maximum feature id and if the file is zero indexed.
     * The maximum feature id is used as part of the padding calculation converting the
     * integer feature numbers into Tribuo's String feature names and is important
     * to set when loading test data to ensure that the names line up with the training
     * names. For example if there are 110 features, but the test dataset only has features
     * 0-90, then without setting {@code maxFeatureID = 110} all the features will be named
     * "00" through "90", rather than the expected "000" - "090", leading to a mismatch.
     * @param path The path to load.
     * @param outputFactory The output factory to use.
     * @param zeroIndexed Are the features in this file indexed from zero?
     * @param maxFeatureID The maximum feature ID allowed.
     * @throws IOException If the file could not be read or is an invalid format.
     */
    public LibSVMDataSource(Path path, OutputFactory<T> outputFactory, boolean zeroIndexed, int maxFeatureID) throws IOException {
        this(path.normalize(),path.normalize().toUri().toURL(),outputFactory,true,zeroIndexed,maxFeatureID);
    }

    /**
     * Constructs a LibSVMDataSource from the supplied URL and output factory.
     * @param url The url to load.
     * @param outputFactory The output factory to use.
     * @throws IOException If the url could not load or is in an invalid format.
     */
    public LibSVMDataSource(URL url, OutputFactory<T> outputFactory) throws IOException {
        this(null,url,outputFactory,false,false,0);
    }

    /**
     * Constructs a LibSVMDataSource from the supplied URL and output factory.
     * <p>
     * Also allows control over the maximum feature id and if the file is zero indexed.
     * The maximum feature id is used as part of the padding calculation converting the
     * integer feature numbers into Tribuo's String feature names and is important
     * to set when loading test data to ensure that the names line up with the training
     * names. For example if there are 110 features, but the test dataset only has features
     * 0-90, then without setting {@code maxFeatureID = 110} all the features will be named
     * "00" through "90", rather than the expected "000" - "090", leading to a mismatch.
     * @param url The url to load.
     * @param outputFactory The output factory to use.
     * @param zeroIndexed Are the features in this file indexed from zero?
     * @param maxFeatureID The maximum feature ID allowed.
     * @throws IOException If the url could not load or is in an invalid format.
     */
    public LibSVMDataSource(URL url, OutputFactory<T> outputFactory, boolean zeroIndexed, int maxFeatureID) throws IOException {
        this(null,url,outputFactory,true,zeroIndexed,maxFeatureID);
    }

    /**
     * Constructs a LibSVMDataSource from the supplied url or path and output factory.
     * <p>
     * One of the url or path must be null.
     * <p>
     * Also allows control over the maximum feature id and if the file is zero indexed.
     * The maximum feature id is used as part of the padding calculation converting the
     * integer feature numbers into Tribuo's String feature names and is important
     * to set when loading test data to ensure that the names line up with the training
     * names. For example if there are 110 features, but the test dataset only has features
     * 0-90, then without setting {@code maxFeatureID = 110} all the features will be named
     * "00" through "90", rather than the expected "000" - "090", leading to a mismatch.
     * @param url The url to load.
     * @param outputFactory The output factory to use.
     * @param zeroIndexed Are the features in this file indexed from zero?
     * @param maxFeatureID The maximum feature ID allowed.
     * @throws IOException If the url could not load or is in an invalid format.
     */
    private LibSVMDataSource(Path path, URL url, OutputFactory<T> outputFactory, boolean rangeSet, boolean zeroIndexed, int maxFeatureID) throws IOException {
        if (url == null && path == null) {
            throw new IllegalArgumentException("Must supply a non-null path or url.");
        }
        this.path = path;
        this.url = url;
        if (outputFactory == null) {
            throw new IllegalArgumentException("outputFactory must not be null");
        }
        this.outputFactory = outputFactory;
        this.rangeSet = rangeSet;
        if (rangeSet) {
            this.zeroIndexed = zeroIndexed;
            this.minFeatureID = zeroIndexed ? 0 : 1;
            if (maxFeatureID < minFeatureID + 1) {
                throw new IllegalArgumentException("maxFeatureID must be positive, found " + maxFeatureID);
            }
            this.maxFeatureID = maxFeatureID;
        }
        read();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        if (maxFeatureID != Integer.MIN_VALUE) {
            rangeSet = true;
            minFeatureID = zeroIndexed ? 0 : 1;
            if (maxFeatureID < minFeatureID + 1) {
                throw new IllegalArgumentException("maxFeatureID must be positive, found " + maxFeatureID);
            }
        }
        if ((url == null) && (path == null)) {
            throw new PropertyException("","path","At most one of url and path must be set.");
        } else if ((url != null) && (path != null) && !path.toUri().toURL().equals(url)) {
            throw new PropertyException("","path","At most one of url and path must be set");
        } else if (path != null) {
            // url is the store of record.
            try {
                url = path.toUri().toURL();
            } catch (MalformedURLException e) {
                throw new PropertyException(e,"","path","Path was not a valid URL");
            }
        }
        read();
    }

    /**
     * Returns true if this dataset is zero indexed, false otherwise (i.e., it starts from 1).
     * @return True if zero indexed.
     */
    public boolean isZeroIndexed() {
        return minFeatureID == 0;
    }

    /**
     * Gets the maximum feature ID found.
     * @return The maximum feature id.
     */
    public int getMaxFeatureID() {
        return maxFeatureID;
    }

    @Override
    public String toString() {
        if (path != null) {
            return "LibSVMDataSource(path=" + path.toString() + ",zeroIndexed="+zeroIndexed+",minFeatureID=" + minFeatureID + ",maxFeatureID=" + maxFeatureID + ")";
        } else {
            return "LibSVMDataSource(url=" + url.toString() + ",zeroIndexed="+zeroIndexed+",minFeatureID=" + minFeatureID + ",maxFeatureID=" + maxFeatureID + ")";
        }
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

    private LibSVMDataSourceProvenance cacheProvenance() {
        return new LibSVMDataSourceProvenance(this);
    }

    private void read() throws IOException {
        int pos = 0;
        ArrayList<HashMap<Integer,Double>> processedData = new ArrayList<>();
        ArrayList<String> labels = new ArrayList<>();

        // Idiom copied from Files.readAllLines,
        // but this doesn't require keeping the whole file in RAM.
        String line;
        // Parse the libsvm file, ignoring malformed lines.
        try (BufferedReader r = new BufferedReader(new InputStreamReader(url.openStream(),StandardCharsets.UTF_8))) {
            for (;;) {
                line = r.readLine();
                if (line == null) {
                    break;
                }
                pos++;
                String[] fields = splitPattern.split(line);
                try {
                    boolean valid = true;
                    HashMap<Integer, Double> features = new HashMap<>();
                    for (int i = 1; i < fields.length && valid; i++) {
                        int ind = fields[i].indexOf(':');
                        if (ind < 0) {
                            logger.warning(String.format("Weird line at %d", pos));
                            valid = false;
                        }
                        String ids = fields[i].substring(0, ind);
                        int id = Integer.parseInt(ids);
                        if ((!rangeSet) && (maxFeatureID < id)) {
                            maxFeatureID = id;
                        }
                        if ((!rangeSet) && (minFeatureID > id)) {
                            minFeatureID = id;
                        }
                        double val = Double.parseDouble(fields[i].substring(ind + 1));
                        Double value = features.put(id, val);
                        if (value != null) {
                            logger.warning(String.format("Repeated features at line %d", pos));
                            valid = false;
                        }
                    }
                    if (valid) {
                        // Store the label
                        labels.add(fields[0]);
                        // Store the features
                        processedData.add(features);
                    } else {
                        throw new IOException("Invalid LibSVM format file");
                    }
                } catch (NumberFormatException ex) {
                    logger.warning(String.format("Weird line at %d", pos));
                    throw new IOException("Invalid LibSVM format file", ex);
                }
            }
        }

        // Calculate the string width
        int width = (""+maxFeatureID).length();
        String formatString = "%0"+width+"d";

        // Check to see if it's zero indexed or one indexed, if we didn't observe the zero feature
        // we assume it's one indexed.
        int maxID = maxFeatureID;
        if (minFeatureID != 0) {
            minFeatureID = 1;
            zeroIndexed = false;
        } else {
            maxID++;
            zeroIndexed = true;
        }

        String[] featureNames = new String[maxID];
        for (int i = 0; i < maxID; i++) {
            featureNames[i] = String.format(formatString,i);
        }

        // Generate examples from the processed data
        ArrayList<Feature> buffer = new ArrayList<>();
        for (int i = 0; i < processedData.size(); i++) {
            String labelStr = labels.get(i);
            HashMap<Integer,Double> features = processedData.get(i);
            try {
                T curLabel = outputFactory.generateOutput(labelStr);
                ArrayExample<T> example = new ArrayExample<>(curLabel);
                buffer.clear();
                for (Map.Entry<Integer, Double> e : features.entrySet()) {
                    // Null check to remove out of range feature indices from test data, if rangeSet was true
                    int id = e.getKey() - minFeatureID;
                    if (id < maxID)  {
                        double value = e.getValue();
                        Feature f = new Feature(featureNames[id], value);
                        buffer.add(f);
                    }
                }
                example.addAll(buffer);
                data.add(example);
            } catch (NumberFormatException e) {
                // If the output isn't a valid number for regression tasks.
                // Features are checked in the input loop above.
                logger.warning(String.format("Failed to parse example %d",i));
                throw new IOException("Invalid LibSVM format file");
            }
        }
    }

    /**
     * The number of examples.
     * @return The number of examples.
     */
    public int size() {
        return data.size();
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return data.iterator();
    }

    /**
     * Writes out a dataset in LibSVM format.
     * <p>
     * Can write either zero indexed or one indexed.
     *
     * @param dataset The dataset to write out.
     * @param out A stream to write it to.
     * @param zeroIndexed If true start the feature numbers from zero, otherwise start from one.
     * @param transformationFunc A function which transforms an {@link Output} into a number.
     * @param <T> The type of the Output.
     */
    public static <T extends Output<T>> void writeLibSVMFormat(Dataset<T> dataset, PrintStream out, boolean zeroIndexed, Function<T,Number> transformationFunc) {
        int modifier = zeroIndexed ? 0 : 1;
        ImmutableFeatureMap featureMap = dataset.getFeatureIDMap();
        for (Example<T> example : dataset) {
            out.print(transformationFunc.apply(example.getOutput()));
            out.print(' ');
            for (Feature feature : example) {
                out.print(featureMap.get(feature.getName()).getID() + modifier);
                out.print(':');
                out.print(feature.getValue());
                out.print(' ');
            }
            out.print('\n');
        }
    }

    /**
     * The provenance for a {@link LibSVMDataSource}.
     */
    public static final class LibSVMDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance fileModifiedTime;
        private final DateTimeProvenance dataSourceCreationTime;
        private final HashProvenance sha256Hash;

        /**
         * Constructs a provenance from the host object's information.
         * @param host The host LibSVMDataSource.
         * @param <T> The output type.
         */
        <T extends Output<T>> LibSVMDataSourceProvenance(LibSVMDataSource<T> host) {
            super(host,"DataSource");
            Optional<OffsetDateTime> time = ProvenanceUtil.getModifiedTime(host.url);
            this.fileModifiedTime = time.map(offsetDateTime -> new DateTimeProvenance(FILE_MODIFIED_TIME, offsetDateTime)).orElseGet(() -> new DateTimeProvenance(FILE_MODIFIED_TIME, OffsetDateTime.MIN));
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE,RESOURCE_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.url));
        }

        /**
         * Constructs a provenance during unmarshalling.
         * @param map The map of unmarshalled provenances.
         */
        public LibSVMDataSourceProvenance(Map<String,Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private LibSVMDataSourceProvenance(ExtractedInfo info) {
            super(info);
            this.fileModifiedTime = (DateTimeProvenance) info.instanceValues.get(FILE_MODIFIED_TIME);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
            this.sha256Hash = (HashProvenance) info.instanceValues.get(RESOURCE_HASH);
        }

        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, LibSVMDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, LibSVMDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String,PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(FILE_MODIFIED_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,FILE_MODIFIED_TIME,DateTimeProvenance.class, LibSVMDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, LibSVMDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(RESOURCE_HASH,ObjectProvenance.checkAndExtractProvenance(configuredParameters,RESOURCE_HASH,HashProvenance.class, LibSVMDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof LibSVMDataSourceProvenance)) return false;
            if (!super.equals(o)) return false;
            LibSVMDataSourceProvenance pairs = (LibSVMDataSourceProvenance) o;
            return fileModifiedTime.equals(pairs.fileModifiedTime) &&
                    dataSourceCreationTime.equals(pairs.dataSourceCreationTime) &&
                    sha256Hash.equals(pairs.sha256Hash);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), fileModifiedTime, dataSourceCreationTime, sha256Hash);
        }

        @Override
        public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
            Map<String,PrimitiveProvenance<?>> map = super.getInstanceValues();

            map.put(FILE_MODIFIED_TIME,fileModifiedTime);
            map.put(DATASOURCE_CREATION_TIME,dataSourceCreationTime);
            map.put(RESOURCE_HASH,sha256Hash);

            return map;
        }
    }

}
