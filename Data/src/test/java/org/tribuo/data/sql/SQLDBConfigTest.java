/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import com.oracle.labs.mlrg.olcut.config.ConfigurationData;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.property.MapProperty;
import com.oracle.labs.mlrg.olcut.config.property.SimpleProperty;
import com.oracle.labs.mlrg.olcut.config.protobuf.ProtoProvenanceSerialization;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.sql.Connection;
import java.sql.Driver;
import java.sql.DriverManager;
import java.sql.DriverPropertyInfo;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SQLDBConfigTest {
    private static final String URL = "jdbc:tribuo-test:credentials";
    private static final String HOST_URL = "jdbc:oracle:thin:@tribuo-test:1521/test";
    private static final String USER = "synthetic-map-user";
    private static final String PASSWORD = "synthetic-map-password";

    static Stream<Function<Map<String, String>, SQLDBConfig>> constructors() {
        return Stream.of(
                properties -> new SQLDBConfig(URL, properties),
                properties -> new SQLDBConfig(URL, null, null, properties),
                properties -> new SQLDBConfig("tribuo-test", "1521", "test", null, null, properties));
    }

    @ParameterizedTest
    @MethodSource("constructors")
    public void credentialsAreRedactedAndReachJdbc(Function<Map<String, String>, SQLDBConfig> constructor) throws SQLException {
        Map<String, String> properties = Map.of("user", USER, "password", PASSWORD, "applicationName", "tribuo-test");
        SQLDBConfig config = constructor.apply(properties);

        assertRedacted(config, USER, PASSWORD);
        assertTrue(config.getProvenance().toString().contains("tribuo-test"));
        assertEquals(properties, connectionProperties(config));
        config.postConfig();
        assertRedacted(config, USER, PASSWORD);
        assertEquals(properties, connectionProperties(config));
    }

    @ParameterizedTest
    @MethodSource("constructors")
    public void callerMapCannotChangeCredentialsOrProvenance(Function<Map<String, String>, SQLDBConfig> constructor) throws SQLException {
        Map<String, String> original = Map.of("user", USER, "password", PASSWORD, "applicationName", "original");
        Map<String, String> properties = new HashMap<>(original);
        SQLDBConfig config = constructor.apply(properties);
        assertEquals(original, properties);

        properties.put("user", "replacement-user");
        properties.put("password", "replacement-password");
        properties.put("applicationName", "replacement-application");

        assertRedacted(config, USER, PASSWORD, "replacement-user", "replacement-password");
        assertEquals(original, connectionProperties(config));
        assertFalse(config.getProvenance().toString().contains("replacement-application"));
    }

    @ParameterizedTest
    @MethodSource("constructors")
    public void singleCredentialsAndEmptyPropertiesStillReachJdbc(Function<Map<String, String>, SQLDBConfig> constructor) throws SQLException {
        for (Map<String, String> properties : List.of(Map.of("user", USER), Map.of("password", PASSWORD), Map.<String, String>of())) {
            SQLDBConfig config = constructor.apply(properties);
            assertRedacted(config, USER, PASSWORD);
            assertEquals(properties, connectionProperties(config));
        }
    }

    @Test
    public void configurationManagerStillNormalizesCredentials() throws Exception {
        Map<String, String> properties = Map.of("user", USER, "password", PASSWORD, "applicationName", "tribuo-test");
        try (ConfigurationManager manager = new ConfigurationManager()) {
            manager.addConfiguration(new ConfigurationData("sql", SQLDBConfig.class.getName(), Map.of(
                    "connectionString", new SimpleProperty(URL),
                    "username", new SimpleProperty("configured-user"),
                    "propMap", MapProperty.createFromStringMap(properties))));
            SQLDBConfig config = (SQLDBConfig) manager.lookup("sql");
            assertRedacted(config, USER, PASSWORD, "configured-user");
            assertEquals(Map.of("user", "configured-user", "password", PASSWORD, "applicationName", "tribuo-test"),
                    connectionProperties(config));
        }
    }

    @Test
    public void explicitCredentialsOverridePropertiesIndividually() throws SQLException {
        Map<String, String> properties = Map.of("user", USER, "password", PASSWORD);
        for (String user : new String[]{null, "explicit-user"}) {
            for (String password : new String[]{null, "explicit-password"}) {
                Map<String, String> expected = Map.of("user", user == null ? USER : user,
                        "password", password == null ? PASSWORD : password);
                for (SQLDBConfig config : new SQLDBConfig[]{
                        new SQLDBConfig(URL, user, password, properties),
                        new SQLDBConfig("tribuo-test", "1521", "test", user, password, properties)}) {
                    assertRedacted(config, USER, PASSWORD, "explicit-user", "explicit-password");
                    assertEquals(expected, connectionProperties(config));
                }
            }
        }
    }

    private static void assertRedacted(SQLDBConfig config, String... secrets) {
        var provenance = config.getProvenance();
        var serialized = new ProtoProvenanceSerialization(false)
                .serializeToProto(ProvenanceUtil.marshalProvenance(provenance));
        for (String secret : secrets) {
            assertFalse(provenance.toString().contains(secret), "Credential in provenance");
            assertFalse(serialized.toString().contains(secret), "Credential in serialized provenance");
        }
    }

    private static Properties connectionProperties(SQLDBConfig config) throws SQLException {
        CapturingDriver driver = new CapturingDriver();
        DriverManager.registerDriver(driver);
        try {
            assertSame(driver.captured, assertThrows(SQLException.class, config::getConnection));
            return driver.properties;
        } finally {
            DriverManager.deregisterDriver(driver);
        }
    }

    /** Captures JDBC properties without connecting to a database. */
    private static final class CapturingDriver implements Driver {
        private final SQLException captured = new SQLException("Captured test connection");
        private Properties properties;

        @Override
        public Connection connect(String url, Properties info) throws SQLException {
            if (!acceptsURL(url)) {
                return null;
            }
            properties = new Properties();
            properties.putAll(info);
            throw captured;
        }

        @Override
        public boolean acceptsURL(String url) {
            return URL.equals(url) || HOST_URL.equals(url);
        }

        @Override
        public DriverPropertyInfo[] getPropertyInfo(String url, Properties info) {
            return new DriverPropertyInfo[0];
        }

        @Override
        public int getMajorVersion() {
            return 1;
        }

        @Override
        public int getMinorVersion() {
            return 0;
        }

        @Override
        public boolean jdbcCompliant() {
            return false;
        }

        @Override
        public Logger getParentLogger() {
            return Logger.getLogger(SQLDBConfigTest.class.getName());
        }
    }
}
