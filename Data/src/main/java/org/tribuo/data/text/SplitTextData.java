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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Splits data in our standard text format into training and testing portions.
 * <p>
 * Checks all the lines are valid before splitting.
 */
public class SplitTextData {
    private static final Logger logger = Logger.getLogger(SplitTextData.class.getName());

    /**
     * Command line options.
     */
    public static class TrainTestSplitOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Splits a standard text format dataset in two.";
        }

        /**
         * Split fraction.
         */
        @Option(charName = 's', longName = "split-fraction", usage = "Split fraction.")
        public float splitFraction;
        /**
         * Input data file in standard text format.
         */
        @Option(charName = 'i', longName = "input-file", usage = "Input data file in standard text format.")
        public Path inputPath;
        /**
         * Output training data file.
         */
        @Option(charName = 't', longName = "training-output-file", usage = "Output training data file.")
        public Path trainPath;
        /**
         * Output validation data file.
         */
        @Option(charName = 'v', longName = "validation-output-file", usage = "Output validation data file.")
        public Path validationPath;
        /**
         * Seed for the RNG.
         */
        @Option(charName = 'r', longName = "rng-seed", usage = "Seed for the RNG.")
        public long seed = 1;
    }

    /**
     * Runs the SplitTextData CLI.
     * @param args The CLI arguments.
     * @throws IOException If the files could not be read or written to.
     */
    public static void main(String[] args) throws IOException {
        
        //
        // Use the labs format logging.
        for (Handler h : Logger.getLogger("").getHandlers()) {
            h.setLevel(Level.ALL);
            h.setFormatter(new LabsLogFormatter());
            try {
                h.setEncoding("utf-8");
            } catch (SecurityException | UnsupportedEncodingException ex) {
                logger.severe("Error setting output encoding");
            }
        }

        TrainTestSplitOptions options = new TrainTestSplitOptions();
        ConfigurationManager cm = new ConfigurationManager(args,options);

        if ((options.inputPath == null) || (options.trainPath == null) || (options.validationPath == null) || (options.splitFraction < 0.0) || (options.splitFraction > 1.0)) {
            System.out.println("Incorrect arguments");
            System.out.println(cm.usage());
            return;
        }

        int n = 0;
        int validCounter = 0;
        int invalidCounter = 0;

        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(options.inputPath.toFile()), StandardCharsets.UTF_8));
        
        PrintWriter trainOutput = new PrintWriter(new OutputStreamWriter(new BufferedOutputStream(new FileOutputStream(options.trainPath.toFile())),StandardCharsets.UTF_8));
        PrintWriter testOutput = new PrintWriter(new OutputStreamWriter(new BufferedOutputStream(new FileOutputStream(options.validationPath.toFile())),StandardCharsets.UTF_8));

        ArrayList<Line> lines = new ArrayList<>();
        while (input.ready()) {
            n++;
            String line = input.readLine().trim();
            if(line.isEmpty()) {
                invalidCounter++;
                continue;
            }
            String[] fields = line.split("##");
            if(fields.length != 2) {
                invalidCounter++;
                logger.warning(String.format("Bad line in %s at %d: %s",
                        options.inputPath, n, line.substring(Math.min(50, line.length()))));
                continue;
            }
            String label = fields[0].trim().toUpperCase();
            lines.add(new Line(label,fields[1]));
            validCounter++;
        }

        input.close();

        logger.info("Found " + validCounter + " valid examples, " + invalidCounter + " invalid examples out of " + n + " lines.");

        int numTraining = Math.round(options.splitFraction * validCounter);
        int numTesting = validCounter - numTraining;

        logger.info("Outputting " + numTraining + " training examples, and " + numTesting + " testing examples, with a " + options.splitFraction + " split.");

        Collections.shuffle(lines,new Random(options.seed));
        for (int i = 0; i < numTraining; i++) {
            trainOutput.println(lines.get(i));
        }
        for (int i = numTraining; i < validCounter; i++) {
            testOutput.println(lines.get(i));
        }

        trainOutput.close();
        testOutput.close();
    }

    private static class Line {
        public final String label;
        public final String text;

        Line(String label, String text) {
            this.label = label;
            this.text = text;
        }

        public String toString() {
            return label + "##" + text;
        }
    }

}

