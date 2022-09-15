/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.data;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.IOUtil;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.MutableDataset;
import org.tribuo.Output;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.GZIPOutputStream;

/**
 * Reads in a Datasource, processes all the data, and writes it out as a serialized dataset. This makes sharing
 * data preprocessing between multiple runs easier.
 */
public final class PreprocessAndSerialize {
    private static final Logger logger = Logger.getLogger(PreprocessAndSerialize.class.getName());

    private PreprocessAndSerialize() {}

    /**
     * Command line options.
     */
    public static class PreprocessAndSerializeOptions implements Options {
        /**
         * Datasource to load from a config file
         */
        @Option(charName = 'd', longName = "dataSource", usage = "Datasource to load from a config file")
        public ConfigurableDataSource<? extends Output<?>> dataSource;
        /**
         * path to serialize the dataset
         */
        @Option(charName = 'o', longName = "serialized-dataset", usage = "path to serialize the dataset")
        public Path output;

        /**
         * Save the dataset as a protobuf.
         */
        @Option(charName = 'p', longName = "save-as-protobuf", usage = "Save the dataset as a protobuf.")
        public boolean protobufFormat;
    }

    /**
     * Run the PreprocessAndSerialize CLI.
     * @param args The CLI args.
     */
    public static void main(String[] args) {

        LabsLogFormatter.setAllLogFormatters();

        PreprocessAndSerializeOptions opts = new PreprocessAndSerializeOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,opts);
        } catch (UsageException e) {
            logger.info(e.getUsage());
            System.exit(1);
        }

        logger.info("Reading datasource into dataset");
        MutableDataset<?> dataset = new MutableDataset<>(opts.dataSource);

        logger.info("Finished reading dataset");

        if(opts.output.endsWith("gz")) {
            logger.info("Writing zipped dataset");
        }
        if (opts.protobufFormat) {
            try (OutputStream os = opts.output.endsWith("gz") ? new GZIPOutputStream(Files.newOutputStream(opts.output)) : Files.newOutputStream(opts.output)) {
                dataset.serializeToStream(os);
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Error writing serialized dataset", e);
            }
        } else {
            try(ObjectOutputStream os = IOUtil.getObjectOutputStream(opts.output.toString(), opts.output.endsWith("gz"))) {
                os.writeObject(dataset);
            } catch (IOException e) {
                logger.log(Level.SEVERE,  "Error writing serialized dataset", e);
                System.exit(1);
            }
        }
    }
}
