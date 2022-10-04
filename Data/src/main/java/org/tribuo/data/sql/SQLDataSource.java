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

package org.tribuo.data.sql;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.data.columnar.ColumnarDataSource;
import org.tribuo.data.columnar.ColumnarIterator;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;

import java.sql.SQLException;
import java.sql.Statement;
import java.time.OffsetDateTime;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A {@link org.tribuo.DataSource} for loading columnar data from a database
 * and applying {@link org.tribuo.data.columnar.FieldProcessor}s to it.
 * The {@link java.sql.Connection}s it creates are closed when the iterator is empty
 * (ie. when hasNext is called and returns false). Calling close() on SQLDatasource itself closes all connections
 * created since close was last called.
 *
 * <p>
 *
 * N.B. This class accepts raw SQL strings and executes them directly via JDBC. It DOES NOT perform
 * any SQL escaping or other injection prevention. It is the user's responsibility to ensure that SQL passed to this
 * class performs as desired.
 */
public class SQLDataSource<T extends Output<T>> extends ColumnarDataSource<T> implements AutoCloseable {

    private static final Logger logger = Logger.getLogger(SQLDataSource.class.getName());

    @Config(mandatory = true,description="Database configuration.")
    private SQLDBConfig sqlConfig;

    @Config(mandatory = true,description="SQL query to run.")
    private String sqlString;

    private final Set<Statement> statements = new HashSet<>();

    /**
     * For OLCUT.
     */
    private SQLDataSource() {}

    /**
     * Constructs a SQLDataSource.
     * @param sqlString The SQL query to run.
     * @param sqlConfig The SQL connection configuration.
     * @param outputFactory The output factory to use.
     * @param rowProcessor The row processor to convert the returned rows into examples.
     * @param outputRequired Is an output required from this source.
     * @throws SQLException If the database could not be read.
     */
    public SQLDataSource(String sqlString, SQLDBConfig sqlConfig, OutputFactory<T> outputFactory, RowProcessor<T> rowProcessor, boolean outputRequired) throws SQLException {
        super(outputFactory, rowProcessor, outputRequired);
        this.sqlConfig = sqlConfig;
        this.sqlString = sqlString;
    }

    @Override
    public String toString() {
        return "SQLDataSource(sqlString=\"" + sqlString + "\", sqlConfig=\"" + sqlConfig.toString() + "\", rowProcessor=" + rowProcessor.getDescription() +")";
    }

    @Override
    public ColumnarIterator rowIterator() {
        try {
            Statement stmt = sqlConfig.getStatement();
            statements.add(stmt);
            return new ResultSetIterator(stmt.executeQuery(sqlString), stmt.getFetchSize());
        } catch (SQLException e) {
            throw new IllegalArgumentException("Error Processing SQL", e);
        }
    }

    @Override
    public void close() {
        for (Statement statement: statements) {
            try {
                statement.close();
            } catch (SQLException e) {
                logger.log(Level.WARNING, "Error closing statement", e);
            }
        }
        statements.clear();
    }

    @Override
    public ConfiguredDataSourceProvenance getProvenance() {
        return new SQLDataSourceProvenance(this);
    }

    /**
     * Provenance for {@link SQLDataSource}.
     */
    public static class SQLDataSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance dataSourceCreationTime;

        <T extends Output<T>> SQLDataSourceProvenance(SQLDataSource<T> host) {
            super(host,"DataSource");
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public SQLDataSourceProvenance(Map<String,Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private SQLDataSourceProvenance(ExtractedInfo info) {
            super(info);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
        }

        /**
         * Separates out the configured and non-configured provenance values.
         * @param map The provenances to separate.
         * @return The extracted provenance information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, SQLDataSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, SQLDataSourceProvenance.class.getSimpleName()).getValue();

            Map<String,PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, SQLDataSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof SQLDataSourceProvenance)) return false;
            if (!super.equals(o)) return false;
            SQLDataSourceProvenance pairs = (SQLDataSourceProvenance) o;
            return dataSourceCreationTime.equals(pairs.dataSourceCreationTime);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), dataSourceCreationTime);
        }

        @Override
        public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
            Map<String,PrimitiveProvenance<?>> map = super.getInstanceValues();

            map.put(DATASOURCE_CREATION_TIME,dataSourceCreationTime);

            return map;
        }

    }
}
