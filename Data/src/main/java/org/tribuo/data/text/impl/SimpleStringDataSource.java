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

import com.oracle.labs.mlrg.olcut.config.Config;
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
import org.tribuo.data.text.TextFeatureExtractor;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;

import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.logging.Logger;

/**
 * A version of {@link SimpleTextDataSource} that accepts a {@link List} of Strings.
 * <p>
 * Uses the parsing logic from {@link SimpleTextDataSource}.
 */
public class SimpleStringDataSource<T extends Output<T>> extends SimpleTextDataSource<T> {

    private static final Logger logger = Logger.getLogger(SimpleStringDataSource.class.getName());

    /**
     * Used because OLCUT doesn't support generic Iterables.
     */
    @Config(mandatory = true,description="The input data lines.")
    protected List<String> rawLines;

    /**
     * For olcut.
     */
    private SimpleStringDataSource() {}

    /**
     * Constructs a simple string data source from the supplied lines.
     * @param rawLines The lines to parse.
     * @param outputFactory The output factory.
     * @param extractor The feature extractor.
     */
    public SimpleStringDataSource(List<String> rawLines, OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor) {
        super(outputFactory, extractor);
        this.rawLines = rawLines;
        this.path = Paths.get(System.getProperty("user.dir"));
        read();
        this.provenance = cacheProvenance();
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        read();
        this.provenance = cacheProvenance();
    }

    @Override
    public String toString() {
        StringBuilder buffer = new StringBuilder();
        buffer.append(this.getClass().getSimpleName());
        buffer.append("(extractor=");
        buffer.append(extractor.toString());
        buffer.append(",preprocessors=");
        buffer.append(preprocessors.toString());
        buffer.append(")");
        return buffer.toString();
    }

    @Override
    protected void read() {
        int n = 0;
        for (String line : rawLines) {
            n++;
            Optional<Example<T>> example = parseLine(line, n);
            example.ifPresent(data::add);
        }
    }

    @Override
    protected ConfiguredDataSourceProvenance cacheProvenance() {
        return new SimpleStringDataSourceProvenance(this);
    }

    /**
     * Provenance for {@link SimpleStringDataSource}.
     */
    public static class SimpleStringDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance dataSourceCreationTime;
        private final HashProvenance sha256Hash;

        <T extends Output<T>> SimpleStringDataSourceProvenance(SimpleStringDataSource<T> host) {
            super(host,"DataSource");
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE,RESOURCE_HASH, ProvenanceUtil.hashList(DEFAULT_HASH_TYPE,host.rawLines));
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public SimpleStringDataSourceProvenance(Map<String, Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private SimpleStringDataSourceProvenance(ExtractedInfo info) {
            super(info);
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
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, SimpleStringDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, SimpleStringDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String, PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, SimpleStringDataSourceProvenance.class.getSimpleName()));
            instanceParameters.put(RESOURCE_HASH,ObjectProvenance.checkAndExtractProvenance(configuredParameters,RESOURCE_HASH,HashProvenance.class, SimpleStringDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof SimpleStringDataSourceProvenance)) return false;
            if (!super.equals(o)) return false;
            SimpleStringDataSourceProvenance pairs = (SimpleStringDataSourceProvenance) o;
            return dataSourceCreationTime.equals(pairs.dataSourceCreationTime) &&
                    sha256Hash.equals(pairs.sha256Hash);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), dataSourceCreationTime, sha256Hash);
        }

        @Override
        public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
            Map<String,PrimitiveProvenance<?>> map = new HashMap<>();

            map.put(DATASOURCE_CREATION_TIME,dataSourceCreationTime);
            map.put(RESOURCE_HASH,sha256Hash);

            return map;
        }
    }

}
