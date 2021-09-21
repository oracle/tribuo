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

package org.tribuo.json;

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
import java.util.logging.Logger;

/**
 * A {@link DataSource} for loading data from a JSON text file
 * and applying {@link FieldProcessor}s to it.
 */
public class JsonDataSource<T extends Output<T>> extends ColumnarDataSource<T> {
    private static final Logger logger = Logger.getLogger(JsonFileIterator.class.getName());

    private URI dataFile;

    @Config(mandatory = true,description="Path to the json file.")
    private Path dataPath;

    private ConfiguredDataSourceProvenance provenance;

    /**
     * For OLCUT.
     */
    private JsonDataSource() {}

    /**
     * Creates a JsonDataSource using the specified RowProcessor to process the data.
     *
     * @param dataPath The Path to the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     */
    public JsonDataSource(Path dataPath, RowProcessor<T> rowProcessor, boolean outputRequired) {
        this(dataPath.toUri(),dataPath,rowProcessor,outputRequired);
    }

    /**
     * Creates a JsonDataSource using the specified RowProcessor to process the data.
     *
     * @param dataFile A URI for the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     */
    public JsonDataSource(URI dataFile, RowProcessor<T> rowProcessor, boolean outputRequired) {
        this(dataFile,Paths.get(dataFile),rowProcessor,outputRequired);
    }

    /**
     * Creates a JsonDataSource using the specified RowProcessor to process the data.
     * @param dataFile A URI for the data file.
     * @param rowProcessor The row processor which converts a row into an {@link Example}.
     * @param outputRequired Is the output required to exist in the data file.
     */
    private JsonDataSource(URI dataFile, Path dataPath, RowProcessor<T> rowProcessor, boolean outputRequired) {
        super(rowProcessor.getResponseProcessor().getOutputFactory(), rowProcessor, outputRequired);
        this.dataPath = dataPath;
        this.dataFile = dataFile;
        this.provenance = new JsonDataSourceProvenance(this);
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        this.dataFile = dataPath.toUri();
        this.provenance = new JsonDataSourceProvenance(this);
    }

    @Override
    public String toString() {
        return "JsonDataSource(file=" + dataFile + ",rowProcessor="+rowProcessor.getDescription()+")";
    }

    @Override
    public ColumnarIterator rowIterator() {
        try {
            return new JsonFileIterator(dataFile);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read data",e);
        }
    }

    @Override
    public ConfiguredDataSourceProvenance getProvenance() {
        return provenance;
    }

    /**
     * Provenance for {@link JsonDataSource}.
     */
    public static class JsonDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance fileModifiedTime;
        private final DateTimeProvenance dataSourceCreationTime;
        private final HashProvenance sha256Hash;

        <T extends Output<T>> JsonDataSourceProvenance(JsonDataSource<T> host) {
            super(host,"DataSource");
            this.fileModifiedTime = new DateTimeProvenance(FILE_MODIFIED_TIME,OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.dataPath.toFile().lastModified()), ZoneId.systemDefault()));
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE,RESOURCE_HASH,ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,host.dataPath));
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public JsonDataSourceProvenance(Map<String,Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private JsonDataSourceProvenance(ExtractedInfo info) {
            super(info);
            this.fileModifiedTime = (DateTimeProvenance) info.instanceValues.get(FILE_MODIFIED_TIME);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
            this.sha256Hash = (HashProvenance) info.instanceValues.get(RESOURCE_HASH);
        }

        /**
         * Splits the provenance into configurable and non-configurable provenances.
         * @param map The provenances to split.
         * @return The extracted information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, JsonDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, JsonDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String,PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(FILE_MODIFIED_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,FILE_MODIFIED_TIME,DateTimeProvenance.class, JsonDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, JsonDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(RESOURCE_HASH,ObjectProvenance.checkAndExtractProvenance(configuredParameters,RESOURCE_HASH,HashProvenance.class, JsonDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            if (!super.equals(o)) return false;
            JsonDataSourceProvenance pairs = (JsonDataSourceProvenance) o;
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
