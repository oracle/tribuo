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

package org.tribuo.data.csv;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.DataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.data.columnar.ColumnarDataSource;
import org.tribuo.data.columnar.ColumnarIterator;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A {@link DataSource} for loading separable data from a text file (e.g. CSV, TSV)
 * and applying {@link FieldProcessor}s to it.
 */
public class CSVDataSource<T extends Output<T>> extends ColumnarDataSource<T> {

    private URI dataFile;

    @Config(mandatory = true,description="Path to the CSV file.")
    private Path dataPath;

    @Config(description="The CSV separator character.")
    private char separator = CSVIterator.SEPARATOR;

    @Config(description="The CSV quote character.")
    private char quote = CSVIterator.QUOTE;

    private ConfiguredDataSourceProvenance provenance;

    /**
     * For OLCUT.
     */
    private CSVDataSource() {}

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data.
     *
     * <p>
     *
     * Uses ',' as the separator, '"' as the quote character, and '\' as the escape character.
     * @param dataPath The Path to the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     */
    public CSVDataSource(Path dataPath, RowProcessor<T> rowProcessor, boolean outputRequired) {
        this(dataPath,rowProcessor,outputRequired, CSVIterator.SEPARATOR, CSVIterator.QUOTE);
    }

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data.
     *
     * <p>
     *
     * Uses ',' as the separator, '"' as the quote character, and '\' as the escape character.
     * @param dataFile A URI for the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     */
    public CSVDataSource(URI dataFile, RowProcessor<T> rowProcessor, boolean outputRequired) {
        this(dataFile,rowProcessor,outputRequired, CSVIterator.SEPARATOR, CSVIterator.QUOTE);
    }

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data.
     *
     * <p>
     *
     * Uses '"' as the quote character, and '\' as the escape character.
     * @param dataPath The Path to the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     * @param separator The separator character in the data file.
     */
    public CSVDataSource(Path dataPath, RowProcessor<T> rowProcessor, boolean outputRequired, char separator) {
        this(dataPath,rowProcessor,outputRequired,separator, CSVIterator.QUOTE);
    }

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data.
     *
     * <p>
     *
     * Uses '"' as the quote character, and '\' as the escape character.
     * @param dataFile A URI for the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     * @param separator The separator character in the data file.
     */
    public CSVDataSource(URI dataFile, RowProcessor<T> rowProcessor, boolean outputRequired, char separator) {
        this(dataFile,rowProcessor,outputRequired,separator, CSVIterator.QUOTE);
    }

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data, and the supplied separator and quote
     * characters to read the input data file.
     * @param dataFile A URI for the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     * @param separator The separator character in the data file.
     * @param quote The quote character in the data file.
     */
    public CSVDataSource(URI dataFile, RowProcessor<T> rowProcessor, boolean outputRequired, char separator, char quote) {
        this(dataFile, Paths.get(dataFile),rowProcessor,outputRequired,separator,quote);
    }

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data, and the supplied separator and quote
     * characters to read the input data file.
     * @param dataPath The Path to the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     * @param separator The separator character in the data file.
     * @param quote The quote character in the data file.
     */
    public CSVDataSource(Path dataPath, RowProcessor<T> rowProcessor, boolean outputRequired, char separator, char quote) {
        this(dataPath.toUri(),dataPath,rowProcessor,outputRequired,separator,quote);
    }

    /**
     * Creates a CSVDataSource using the specified RowProcessor to process the data, and the supplied separator, quote
     * characters to read the input data file.
     * @param dataFile A URI for the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     * @param separator The separator character in the data file.
     * @param quote The quote character in the data file.
     */
    private CSVDataSource(URI dataFile, Path dataPath, RowProcessor<T> rowProcessor, boolean outputRequired, char separator, char quote) {
        super(rowProcessor.getResponseProcessor().getOutputFactory(), rowProcessor, outputRequired);
        this.dataPath = dataPath;
        this.dataFile = dataFile;
        this.separator = separator;
        this.quote = quote;
        this.provenance = new CSVDataSourceProvenance(this);
    }

    @Override
    public void postConfig() {
        this.dataFile = dataPath.toUri();
        this.provenance = new CSVDataSourceProvenance(this);
    }

    @Override
    public String toString() {
        return "CSVDataSource(file=" + dataFile + ",rowProcessor="+rowProcessor.getDescription()+")";
    }

    @Override
    public ColumnarIterator rowIterator() {
        try {
            return new CSVIterator(dataFile, separator, quote);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read data",e);
        }
    }

    @Override
    public ConfiguredDataSourceProvenance getProvenance() {
        return provenance;
    }

    public static class CSVDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance fileModifiedTime;
        private final DateTimeProvenance dataSourceCreationTime;
        private final HashProvenance sha256Hash;

        <T extends Output<T>> CSVDataSourceProvenance(CSVDataSource<T> host) {
            super(host,"DataSource");
            this.fileModifiedTime = new DateTimeProvenance(FILE_MODIFIED_TIME,OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.dataPath.toFile().lastModified()), ZoneId.systemDefault()));
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE,RESOURCE_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.dataPath));
        }

        public CSVDataSourceProvenance(Map<String,Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private CSVDataSourceProvenance(ExtractedInfo info) {
            super(info);
            this.fileModifiedTime = (DateTimeProvenance) info.instanceValues.get(FILE_MODIFIED_TIME);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
            this.sha256Hash = (HashProvenance) info.instanceValues.get(RESOURCE_HASH);
        }

        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, CSVDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, CSVDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String,PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(FILE_MODIFIED_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,FILE_MODIFIED_TIME,DateTimeProvenance.class, CSVDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, CSVDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(RESOURCE_HASH,ObjectProvenance.checkAndExtractProvenance(configuredParameters,RESOURCE_HASH,HashProvenance.class, CSVDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;
            CSVDataSourceProvenance pairs = (CSVDataSourceProvenance) o;
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
