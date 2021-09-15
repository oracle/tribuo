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

import com.opencsv.CSVParserWriter;
import com.opencsv.ICSVWriter;
import com.opencsv.RFC4180Parser;
import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Read an SQL query in on the standard input, write a CSV file containing the
 * results to the standard output.
 * <p>
 * N.B. This class accepts raw SQL strings and executes them directly via JDBC. It DOES NOT perform
 * any SQL escaping or other injection prevention. It is the user's responsibility to ensure that SQL passed to this
 * class performs as desired.
 */
public class SQLToCSV {

    /**
     * Command line options.
     */
    public static class SQLToCSVOptions implements Options {
        /**
         * Connection string to the SQL database
         */
        @Option(charName = 'n', longName = "connection", usage = "Connection string to the SQL database")
        public String connString;
        /**
         * Password for the SQL database
         */
        @Option(charName = 'p', longName = "password", usage = "Password for the SQL database")
        public String password;
        /**
         * Username for the SQL database
         */
        @Option(charName = 'u', longName = "username", usage = "Username for the SQL database")
        public String username;
        /**
         * SQL File to run as a query, defaults to stdin
         */
        @Option(charName = 'i', longName = "input-sql", usage = "SQL File to run as a query, defaults to stdin")
        public Path inputPath;
        /**
         * File to write query results as CSV, defaults to stdout
         */
        @Option(charName = 'o', longName = "output-csv", usage = "File to write query results as CSV, defaults to stdout")
        public Path outputPath;
        /**
         * Name of the DBConfig to use
         */
        @Option(longName = "db-config", usage = "Name of the DBConfig to use")
        public SQLDBConfig dbConfig;

    }

    private static final Logger logger = Logger.getLogger(SQLToCSV.class.getName());

    /**
     * Reads an SQL query from the standard input and writes the results of the
     * query to the standard output.
     *
     * @param args Single arg is the JDBC connection string.
     */
    public static void main(String[] args) {

        LabsLogFormatter.setAllLogFormatters();

        SQLToCSVOptions opts = new SQLToCSVOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,opts);
        } catch (UsageException e) {
            logger.info(e.getUsage());
            System.exit(1);
        }

        if (opts.dbConfig == null) {
            if (opts.connString == null) {
                logger.log(Level.SEVERE, "Must specify connection string with -n");
                System.exit(1);
            }

            if (opts.username != null || opts.password != null) {
                if (opts.username == null || opts.password == null) {
                    logger.log(Level.SEVERE, "Must specify both of user and password with -u, -p if one is specified!");
                    System.exit(1);
                }
            }
        } else if(opts.username != null || opts.password != null || opts.connString != null) {
            logger.warning("dbConfig provided but username/password/connstring also provided. Options from -u, -p, -n being ignored");
        }


        String query;
        try (BufferedReader br = opts.inputPath != null ? Files.newBufferedReader(opts.inputPath) : new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            StringBuilder qsb = new StringBuilder();
            String l;
            while ((l = br.readLine()) != null) {
                qsb.append(l);
                qsb.append("\n");
            }
            query = qsb.toString().trim();
        } catch (IOException ex) {
            logger.log(Level.SEVERE, "Error reading query: " + ex);
            System.exit(1);
            return;
        }

        if (query.isEmpty()) {
            logger.log(Level.SEVERE, "Query is empty string");
            System.exit(1);
        }

        Connection conn = null;
        try {
            if (opts.dbConfig != null) {
                conn = opts.dbConfig.getConnection();
            } else if (opts.username != null) {
                conn = DriverManager.getConnection(opts.connString, opts.username, opts.password);
            } else {
                conn = DriverManager.getConnection(opts.connString);
            }
        } catch (SQLException ex) {
            logger.log(Level.SEVERE, "Can't connect to database: " + opts.connString, ex);
            System.exit(1);
        }


        try (Statement stmt = conn.createStatement()){
            stmt.setFetchSize(1000);
            stmt.setFetchDirection(ResultSet.FETCH_FORWARD);

            ResultSet results;
            try {
                results = stmt.executeQuery(query);
            } catch (SQLException ex) {
                logger.log(Level.SEVERE, "Error running query", ex);
                try {
                    conn.close();
                } catch (SQLException ex1) {
                    logger.log(Level.SEVERE, "Failed to close connection", ex1);
                }
                return;
            }

            try(ICSVWriter writer = new CSVParserWriter(opts.outputPath != null ?
                    Files.newBufferedWriter(opts.outputPath) :
                    new BufferedWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8), 1024 * 1024), new RFC4180Parser(), "\n")) {
                writer.writeAll(results, true);
            } catch (IOException ex) {
                logger.log(Level.SEVERE, "Error writing CSV", ex);
                System.exit(1);
            } catch (SQLException ex) {
                logger.log(Level.SEVERE, "Error retrieving results", ex);
                System.exit(1);
            }
        } catch (SQLException ex) {
            logger.log(Level.SEVERE, "Couldn't create statement", ex);
            try {
                conn.close();
            } catch (SQLException ex1) {
                logger.log(Level.SEVERE, "Failed to close connection", ex1);
            }
            System.exit(1);
            return;
        }

        try {
            conn.close();
        } catch (SQLException ex1) {
            logger.log(Level.SEVERE, "Failed to close connection", ex1);
        }

    }

}
