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

package org.tribuo.classification.sequence;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import com.oracle.labs.mlrg.olcut.util.LabsLogFormatter;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.hash.MessageDigestHasher;
import org.tribuo.impl.ArrayExample;
import org.tribuo.sequence.ImmutableSequenceDataset;
import org.tribuo.sequence.MutableSequenceDataset;
import org.tribuo.sequence.SequenceDataset;
import org.tribuo.sequence.SequenceExample;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Path;
import java.util.logging.Logger;

/**
 * Hashes all the features in a sequence dataset.
 */
public class Anonymise {
    private static final Logger logger = Logger.getLogger(AnonymiseOptions.class.getName());

    public enum AnonType { HASH, NUMBER }

    public static class AnonymiseOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "Anonymises a dataset by hashing all it's features and labels.";
        }
        @Option(charName='s',longName="input-train-dataset",usage="Path to the input serialised training SequenceDataset.")
        public Path inputTrainDataset = null;
        @Option(charName='t',longName="input-test-dataset",usage="Path to the input serialised testing SequenceDataset.")
        public Path inputTestDataset = null;
        @Option(charName='u',longName="output-train-dataset",usage="Path to output the serialised training SequenceDataset.")
        public Path outputTrainDataset = null;
        @Option(charName='v',longName="output-test-dataset",usage="Path to output the serialised testing SequenceDataset.")
        public Path outputTestDataset = null;
        @Option(charName='a',longName="anonymisation-type",usage="Method of anonymisation. Defaults to NUMBER.")
        public AnonType type = AnonType.NUMBER;
    }

    public static MutableSequenceDataset<Label> transformDataset(SequenceDataset<Label> input, ImmutableFeatureMap fMap, ImmutableOutputInfo<Label> outputInfo){
        int length = (""+fMap.size()).length();
        int labelLength = (""+outputInfo.size()).length();
        String featureFormatString = "%"+length+"d";
        String labelFormatString = "%"+labelLength+"d";

        MutableSequenceDataset<Label> output = new MutableSequenceDataset<>(input.getSourceProvenance(),input.getOutputFactory());
        for (SequenceExample<Label> se : input) {
            SequenceExample<Label> transformedSequenceExample = new SequenceExample<>();
            for (Example<Label> e : se) {
                int labelId = outputInfo.getID(e.getOutput());
                String newLabelString = String.format(labelFormatString,labelId);
                Label newLabel = new Label(newLabelString);
                ArrayExample<Label> newE = new ArrayExample<>(newLabel);
                for (Feature f : e) {
                    Feature transformedFeature = new Feature(String.format(featureFormatString,fMap.getID(f.getName())),f.getValue());
                    newE.add(transformedFeature);
                }
                transformedSequenceExample.addExample(newE);
            }
            output.add(transformedSequenceExample);
        }
        return output;
    }

    public static MutableSequenceDataset<Label> hashDataset(SequenceDataset<Label> input, ImmutableFeatureMap fMap, ImmutableOutputInfo<Label> outputInfo){
        MessageDigestHasher hasher = new MessageDigestHasher("SHA-256","1234567890");
        MutableSequenceDataset<Label> output = new MutableSequenceDataset<>(input.getSourceProvenance(),input.getOutputFactory());
        for (SequenceExample<Label> se : input) {
            SequenceExample<Label> transformedSequenceExample = new SequenceExample<>();
            for (Example<Label> e : se) {
                String newLabelString = hasher.hash(e.getOutput().getLabel());
                Label newLabel = new Label(newLabelString);
                ArrayExample<Label> newE = new ArrayExample<>(newLabel);
                for (Feature f : e) {
                    Feature transformedFeature = new Feature(hasher.hash(f.getName()),f.getValue());
                    newE.add(transformedFeature);
                }
                transformedSequenceExample.addExample(newE);
            }
            output.add(transformedSequenceExample);
        }
        return output;
    }

    @SuppressWarnings("unchecked") // deserialising a sequence dataset.
    public static void main(String[] args) throws ClassNotFoundException, IOException {
        //
        // Use the labs format logging.
        LabsLogFormatter.setAllLogFormatters();

        AnonymiseOptions o = new AnonymiseOptions();
        ConfigurationManager cm;
        try {
            cm = new ConfigurationManager(args,o);
        } catch (UsageException e) {
            logger.info(e.getMessage());
            return;
        }

        if ((o.inputTrainDataset == null) || (o.inputTestDataset == null) || (o.outputTestDataset == null) || (o.outputTrainDataset == null)) {
            logger.info("No input and output datasets provided.");
            logger.info(cm.usage());
            return;
        }

        // Load the input data
        SequenceDataset<Label> train;
        SequenceDataset<Label> test;
        ImmutableFeatureMap trainingFeatureMap;
        ImmutableOutputInfo<Label> trainingOutputMap;
        logger.info("Loading training data from " + o.inputTrainDataset);
        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(o.inputTrainDataset.toFile())));
             ObjectInputStream oits = new ObjectInputStream(new BufferedInputStream(new FileInputStream(o.inputTestDataset.toFile())))) {
            train = (SequenceDataset<Label>) ois.readObject();
            logger.info(String.format("Loaded %d training examples for %s", train.size(), train.getOutputs().toString()));
            logger.info("Found " + train.getFeatureIDMap().size() + " features");
            logger.info("Loading testing data from " + o.inputTestDataset);
            SequenceDataset<Label> deserTest = (SequenceDataset<Label>) oits.readObject();
            trainingFeatureMap = train.getFeatureIDMap();
            trainingOutputMap = train.getOutputIDInfo();
            test = ImmutableSequenceDataset.copyDataset(deserTest,trainingFeatureMap,trainingOutputMap);
            logger.info(String.format("Loaded %d testing examples", test.size()));
        }

        // Anonymise the datasets.
        logger.info("Transforming the data using " + o.type);
        MutableSequenceDataset<Label> transformedTrain;
        MutableSequenceDataset<Label> transformedTest;
        switch (o.type) {
            case NUMBER:
                transformedTrain = transformDataset(train,trainingFeatureMap,trainingOutputMap);
                transformedTest = transformDataset(test,trainingFeatureMap,trainingOutputMap);
                break;
            case HASH:
                transformedTrain = hashDataset(train,trainingFeatureMap,trainingOutputMap);
                transformedTest = hashDataset(test,trainingFeatureMap,trainingOutputMap);
                break;
            default:
                throw new IllegalArgumentException("Unknown transform type " + o.type);
        }

        // Write out the output data
        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(o.outputTrainDataset.toFile())));
             ObjectOutputStream oots = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(o.outputTestDataset.toFile())))) {
            logger.info("Writing out training data to " + o.outputTrainDataset);
            oos.writeObject(transformedTrain);
            logger.info("Writing out testing data to " + o.outputTestDataset);
            oots.writeObject(transformedTest);
        }
    }
}
