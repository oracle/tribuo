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

package org.tribuo.util.infotheory.example;

import com.oracle.labs.mlrg.olcut.config.ConfigurationManager;
import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import com.oracle.labs.mlrg.olcut.config.UsageException;
import org.tribuo.util.infotheory.InformationTheory;
import org.tribuo.util.infotheory.impl.CachedTriple;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Demo showing how to calculate various mutual informations and entropies.
 */
public class InformationTheoryDemo {

    private static final Logger logger = Logger.getLogger(InformationTheoryDemo.class.getName());

    private static final Random rng = new Random(1);

    /**
     * Generates a sample from a uniform distribution over the integers.
     * @param length The number of samples.
     * @param alphabetSize The alphabet size (i.e., the number of unique values).
     * @return A sample from a uniform distribution.
     */
    public static List<Integer> generateUniform(int length, int alphabetSize) {
        List<Integer> vector = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            vector.add(i,rng.nextInt(alphabetSize));
        }

        return vector;
    }

    /**
     * Generates a sample from a three variable XOR function.
     * <p>
     * Each list is a binary variable, and the third is the XOR of the first two.
     * @param length The number of samples.
     * @return A sample from an XOR function.
     */
    public static CachedTriple<List<Integer>,List<Integer>,List<Integer>> generateXOR(int length) {
        List<Integer> first = new ArrayList<>(length);
        List<Integer> second = new ArrayList<>(length);
        List<Integer> xor = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            int firstVal = rng.nextInt(2);
            int secondVal = rng.nextInt(2);
            int xorVal = firstVal ^ secondVal;
            first.add(i,firstVal);
            second.add(i,secondVal);
            xor.add(i,xorVal);
        }

        return new CachedTriple<>(first,second,xor);
    }

    /**
     * These correlations don't map to mutual information values, as if xyDraw is above xyCorrelation then the draw is completely random.
     * <p>
     * To make it generate correlations of a specific mutual information then it needs to specify the full joint distribution and draw from that.
     * @param length The number of samples.
     * @param alphabetSize The alphabet size (i.e., the number of unique values).
     * @param xyCorrelation Value between 0.0 and 1.0 specifying how likely it is that Y has the same value as X.
     * @param xzCorrelation Value between 0.0 and 1.0 specifying how likely it is that Z has the same value as X.
     * @return A triple of samples drawn from correlated random variables.
     */
    public static CachedTriple<List<Integer>,List<Integer>,List<Integer>> generateCorrelated(int length, int alphabetSize, double xyCorrelation, double xzCorrelation) {
        List<Integer> first = new ArrayList<>(length);
        List<Integer> second = new ArrayList<>(length);
        List<Integer> third = new ArrayList<>(length);

        for (int i = 0; i < length; i++) {
            int firstVal = rng.nextInt(alphabetSize);
            first.add(firstVal);

            double xyDraw = rng.nextDouble();
            if (xyDraw < xyCorrelation) {
                second.add(firstVal);
            } else {
                second.add(rng.nextInt(alphabetSize));
            }

            double xzDraw = rng.nextDouble();
            if (xzDraw < xzCorrelation) {
                third.add(firstVal);
            } else {
                third.add(rng.nextInt(alphabetSize));
            }
        }

        return new CachedTriple<>(first,second,third);
    }

    /**
     * Type of data distribution.
     */
    public enum DistributionType {
        /**
         * Uniformly randomly generated data.
         */
        RANDOM,
        /**
         * Data generated from an XOR function.
         */
        XOR,
        /**
         * Correlated data.
         */
        CORRELATED
    }

    /**
     * Command line options.
     */
    public static class DemoOptions implements Options {
        @Override
        public String getOptionsDescription() {
            return "A demo class showing how to calculate various mutual informations from different inputs.";
        }

        /**
         * The type of the input distribution.
         */
        @Option(charName = 't', longName = "type", usage = "The type of the input distribution.")
        public DistributionType type = DistributionType.RANDOM;
    }

    /**
     * Runs a simple demo of the information theory functions.
     * @param args The CLI arguments.
     */
    public static void main(String[] args) {

        DemoOptions options = new DemoOptions();

        try {
            ConfigurationManager cm = new ConfigurationManager(args, options, false);
        } catch (UsageException e) {
            System.out.println(e.getUsage());
        }

        List<Integer> x;
        List<Integer> y;
        List<Integer> z;

        switch (options.type) {
            case RANDOM:
                x = generateUniform(1000, 5);
                y = generateUniform(1000, 5);
                z = generateUniform(1000, 5);
                break;
            case XOR:
                CachedTriple<List<Integer>,List<Integer>,List<Integer>> trip = generateXOR(1000);
                x = trip.getA();
                y = trip.getB();
                z = trip.getC();
                break;
            case CORRELATED:
                CachedTriple<List<Integer>,List<Integer>,List<Integer>> tripC = generateCorrelated(1000,5,0.7,0.5);
                x = tripC.getA();
                y = tripC.getB();
                z = tripC.getC();
                break;
            default:
                logger.log(Level.WARNING, "Unknown test case, exiting");
                return;
        }

        double hx = InformationTheory.entropy(x);
        double hy = InformationTheory.entropy(y);
        double hz = InformationTheory.entropy(z);

        double hxy = InformationTheory.jointEntropy(x,y);
        double hxz = InformationTheory.jointEntropy(x,z);
        double hyz = InformationTheory.jointEntropy(y,z);
        
        double ixy = InformationTheory.mi(x,y);
        double ixz = InformationTheory.mi(x,z);
        double iyz = InformationTheory.mi(y,z);
        
        InformationTheory.GTestStatistics gxy = InformationTheory.gTest(x,y,null);
        InformationTheory.GTestStatistics gxz = InformationTheory.gTest(x,z,null);
        InformationTheory.GTestStatistics gyz = InformationTheory.gTest(y,z,null);

        if (InformationTheory.LOG_BASE == InformationTheory.LOG_2) {
            logger.log(Level.INFO, "Using log_2");
        } else if (InformationTheory.LOG_BASE == InformationTheory.LOG_E) {
            logger.log(Level.INFO, "Using log_e");
        } else {
            logger.log(Level.INFO, "Using unexpected log base, LOG_BASE = " + InformationTheory.LOG_BASE);
        }
        
        logger.log(Level.INFO, "The entropy of X, H(X) is " + hx);
        logger.log(Level.INFO, "The entropy of Y, H(Y) is " + hy);
        logger.log(Level.INFO, "The entropy of Z, H(Z) is " + hz);
        
        logger.log(Level.INFO, "The joint entropy of X and Y, H(X,Y) is " + hxy);
        logger.log(Level.INFO, "The joint entropy of X and Z, H(X,Z) is " + hxz);
        logger.log(Level.INFO, "The joint entropy of Y and Z, H(Y,Z) is " + hyz);

        logger.log(Level.INFO, "The mutual information between X and Y, I(X;Y) is " + ixy);
        logger.log(Level.INFO, "The mutual information between X and Z, I(X;Z) is " + ixz);
        logger.log(Level.INFO, "The mutual information between Y and Z, I(Y;Z) is " + iyz);

        logger.log(Level.INFO, "The G-Test between X and Y, G(X;Y) is " + gxy);
        logger.log(Level.INFO, "The G-Test between X and Z, G(X;Z) is " + gxz);
        logger.log(Level.INFO, "The G-Test between Y and Z, G(Y;Z) is " + gyz);
    }
    
}
