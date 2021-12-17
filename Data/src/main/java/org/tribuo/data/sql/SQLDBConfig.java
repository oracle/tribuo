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
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * N.B. This class accepts raw SQL strings and executes them directly via JDBC. It DOES NOT perform
 * any SQL escaping or other injection prevention. It is the user's responsibility to ensure that SQL passed to this
 * class performs as desired.
 * <p>
 * SQL database configuration. If you specify the {@linkplain SQLDBConfig#host}, {@linkplain SQLDBConfig#port}, and
 * {@linkplain SQLDBConfig#db} strings and use {@code oracle.jdbc.OracleDriver} as your JDBC Driver, then this will
 * automatically generate a connectionString, otherwise it must be specified manually and host, port, and db fields
 * can be omitted.
 * <p>
 * {@link java.sql.DriverManager}'s default logic will be used to determine which {@link java.sql.Driver} to use for
 * a given connection string.
 */
public class SQLDBConfig implements Configurable, Provenancable<ConfiguredObjectProvenance> {

    @Config(description="Connection string, including host, port and db.")
    private String connectionString;
    @Config(description="Database username.",redact=true)
    private String username;
    @Config(description="Database password.",redact=true)
    private String password;

    @Config(description="Properties to pass to java.sql.DriverManager, username and password will be removed and populated to their fields. If specified both on the map and in the fields, the fields will be used")
    private Map<String, String> propMap = new HashMap<>();

    @Config(description="Hostname of the database machine.")
    private String host;
    @Config(description="Port number.")
    private String port;
    @Config(description="Database name.")
    private String db;

    @Config(description="Size of batches to fetch from DB for queries")
    private int fetchSize = 1000;

    private SQLDBConfig() {}

    /**
     * Constructs a SQL database configuration.
     * <p>
     * Note it is recommended that wallet based connections are used rather than this constructor using {@link #SQLDBConfig(String,Map)}.
     * @param connectionString The connection string.
     * @param username The username.
     * @param password The password.
     * @param properties The connection properties.
     */
    public SQLDBConfig(String connectionString, String username, String password, Map<String, String> properties) {
        this(connectionString, properties);
        this.username = username;
        this.password = password;
    }

    /**
     * Constructs a SQL database configuration.
     * <p>
     * Note it is recommended that wallet based connections are used rather than this constructor using {@link #SQLDBConfig(String,Map)}.
     * @param host The host to connect to.
     * @param port The port to connect on.
     * @param db The db name.
     * @param username The username.
     * @param password The password.
     * @param properties The connection properties.
     */
    public SQLDBConfig(String host, String port, String db, String username, String password, Map<String, String> properties) {
        this(makeConnectionString(host, port, db), properties);
        this.host = host;
        this.port = port;
        this.db = db;
        this.username = username;
        this.password = password;
    }

    /**
     * Constructs a SQL database configuration.
     * @param connectionString The connection string.
     * @param properties The connection properties.
     */
    public SQLDBConfig(String connectionString, Map<String, String> properties) {
        this.connectionString = connectionString;
        this.propMap = properties;
    }

    private static String makeConnectionString(String host, String port, String db) {
        return "jdbc:oracle:thin:@" + host + ":" + port + "/" + db;
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {

        if(propMap.containsKey("user")) {
            if(username == null) {
                username = propMap.remove("user");
            } else {
                propMap.remove("user");
            }
        }
        if(propMap.containsKey("password")) {
            if(password == null) {
                password = propMap.remove("password");
            } else {
                propMap.remove("password");
            }
        }
        if(connectionString == null) {
            if(host != null && port != null && db != null) {
                connectionString = makeConnectionString(host, port, db);
            } else {
                throw new PropertyException(SQLDBConfig.class.getName(), "connectionString", "All of host, port, and db must be specified if connectionString is null");
            }
        }
    }

    /**
     * Constructs a connection based on the object fields.
     * @return A connection to the database.
     * @throws SQLException If the connection failed.
     */
    public Connection getConnection() throws SQLException {

        Properties props = new Properties();
        props.putAll(propMap);
        if(username != null && password != null) {
            props.put("user", username);
            props.put("password", password);
        }
        return DriverManager.getConnection(connectionString, props);
    }

    /**
     * Constructs a statement based on the object fields. Uses fetchSize to determine fetch size and sets defaults
     * for querying data.
     *
     * @return A statement object for querying the database.
     * @throws SQLException If the connection failed.
     */
    public Statement getStatement() throws SQLException {
        Statement stmt = getConnection().createStatement();
        stmt.setFetchSize(fetchSize);
        stmt.setFetchDirection(ResultSet.FETCH_FORWARD);
        return stmt;
    }

    @Override
    public String toString() {
        if (connectionString != null) {
            return "SQLDBConfig(connectionString="+connectionString+")";
        } else {
            return "SQLDBConfig(host="+host+",port="+port+",db="+db+")";
        }
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"SQL-DB-Config");
    }
}
