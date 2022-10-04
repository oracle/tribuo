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

package org.tribuo.data.text.impl;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.data.text.DocumentPreprocessor;
import org.tribuo.data.text.TextDataSource;
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * A dataset for a simple data format for text classification experiments. A line
 * in the file looks like:
 *
 * <pre>
 * OUTPUT##Document text
 * </pre>
 *
 * Each line in the file specifies a single output and document pair. Leading and
 * trailing spaces will be trimmed from outputs and documents. Outputs will be
 * converted to upper case.
 * 
 * <p> 
 * 
 * As with all of our text data, the file should be in UTF-8.
 */
public class SimpleTextDataSource<T extends Output<T>> extends TextDataSource<T> {

    private static final Logger logger = Logger.getLogger(SimpleTextDataSource.class.getName());

    private static final Pattern splitPattern = Pattern.compile("##");

    /**
     * The data source provenance.
     */
    protected ConfiguredDataSourceProvenance provenance;

    /**
     * for olcut
     */
    protected SimpleTextDataSource() {}

    /**
     * Constructs a simple text data source by reading lines from the supplied path.
     * @param path The path to load.
     * @param outputFactory The output factory to use.
     * @param extractor The feature extractor.
     * @throws IOException If the path could not be read.
     */
    public SimpleTextDataSource(Path path, OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor) throws IOException {
        super(path, outputFactory, extractor);
        postConfig();
    }

    /**
     * Constructs a simple text data source by reading lines from the supplied file.
     * @param file The file to load.
     * @param outputFactory The output factory to use.
     * @param extractor The feature extractor.
     * @throws IOException If the file could not be read.
     */
    public SimpleTextDataSource(File file, OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor) throws IOException {
        super(file, outputFactory, extractor);
        postConfig();
    }

    /**
     * Cosntructs a data source without a path.
     * @param outputFactory The output factory.
     * @param extractor The text extraction pipeline.
     */
    protected SimpleTextDataSource(OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor) {
        super((Path)null,outputFactory,extractor);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() throws IOException {
        read();
        provenance = cacheProvenance();
    }

    /**
     * Parses a line in Tribuo's default text format.
     * @param line The line to parse.
     * @param n The current line number.
     * @return An example or an empty optional if it failed to parse.
     */
    protected Optional<Example<T>> parseLine(String line, int n) {
        line = line.trim();
        if(line.isEmpty()) {
            return Optional.empty();
        }
        String[] fields = splitPattern.split(line);
        if(fields.length != 2) {
            logger.warning(String.format("Bad line in %s at %d: %s",
                    path, n, line.substring(Math.min(50, line.length()))));
            return Optional.empty();
        }
        String document = fields[1];
        for (DocumentPreprocessor preproc : preprocessors) {
            document = preproc.processDoc(document);
            if (document == null) {
                // We processed the document away
                return Optional.empty();
            }
        }
        T label = outputFactory.generateOutput(fields[0].trim().toUpperCase());
        return Optional.of(extractor.extract(label, handleDoc(fields[1].trim())));
    }

    @Override
    protected void read() throws IOException {
        int n = 0;
        for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
            n++;
            Optional<Example<T>> example = parseLine(line, n);
            if (example.isPresent()) {
                Example<T> e = example.get();
                if (e.validateExample()) {
                    data.add(e);
                } else {
                    logger.warning("Invalid example found after parsing line " + n);
                }
            }
        }
    }

    @Override
    public ConfiguredDataSourceProvenance getProvenance() {
        return provenance;
    }

    /**
     * Computes the provenance.
     * @return The provenance.
     */
    protected ConfiguredDataSourceProvenance cacheProvenance() {
        return new SimpleTextDataSourceProvenance(this);
    }

    /**
     * Provenance for {@link SimpleTextDataSource}.
     */
    public static class SimpleTextDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance fileModifiedTime;
        private final DateTimeProvenance dataSourceCreationTime;
        private final HashProvenance sha256Hash;

        <T extends Output<T>> SimpleTextDataSourceProvenance(SimpleTextDataSource<T> host) {
            super(host,"DataSource");
            this.fileModifiedTime = new DateTimeProvenance(FILE_MODIFIED_TIME,OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.path.toFile().lastModified()), ZoneId.systemDefault()));
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE,RESOURCE_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.path));
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public SimpleTextDataSourceProvenance(Map<String,Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private SimpleTextDataSourceProvenance(ExtractedInfo info) {
            super(info);
            this.fileModifiedTime = (DateTimeProvenance) info.instanceValues.get(FILE_MODIFIED_TIME);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
            this.sha256Hash = (HashProvenance) info.instanceValues.get(RESOURCE_HASH);
        }

        /**
         * Separates out the configured and non-configured provenance values.
         * @param map The provenances to separate.
         * @return The extracted provenance information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, SimpleTextDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, SimpleTextDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String,PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(FILE_MODIFIED_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,FILE_MODIFIED_TIME,DateTimeProvenance.class, SimpleTextDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, SimpleTextDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(RESOURCE_HASH,ObjectProvenance.checkAndExtractProvenance(configuredParameters,RESOURCE_HASH,HashProvenance.class, SimpleTextDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof SimpleTextDataSourceProvenance)) return false;
            if (!super.equals(o)) return false;
            SimpleTextDataSourceProvenance pairs = (SimpleTextDataSourceProvenance) o;
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
            Map<String,PrimitiveProvenance<?>> map = new HashMap<>();

            map.put(FILE_MODIFIED_TIME,fileModifiedTime);
            map.put(DATASOURCE_CREATION_TIME,dataSourceCreationTime);
            map.put(RESOURCE_HASH,sha256Hash);

            return map;
        }
    }

}
