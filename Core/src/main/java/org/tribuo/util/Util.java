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

package org.tribuo.util;

import com.oracle.labs.mlrg.olcut.util.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.ToIntFunction;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Ye olde util class.
 * <p>
 * Basically full of vector and RNG operations.
 */
public final class Util {

    private static final Logger logger = Logger.getLogger(Util.class.getName());

    // private constructor of a final class, this is full of static methods so you can't instantiate it.
    private Util() {}

    /**
     * Find the index of the maximum value in a list.
     * @param values list
     * @param <T> the type of the values (must implement Comparable)
     * @return a pair: (index of the max value, max value)
     */
    public static <T extends Comparable<T>> Pair<Integer, T> argmax(List<T> values) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("argmax on an empty list");
        }
        //
        // There is no "globally min" value like -Inf for an arbitrary type T so we just pick the first list element
        T vmax = values.get(0);
        int imax = 0;
        for (int i = 1; i < values.size(); i++) {
            T v = values.get(i);
            if (v.compareTo(vmax) > 0) {
                vmax = v;
                imax = i;
            }
        }
        return new Pair<>(imax, vmax);
    }

    /**
     * Find the index of the minimum value in a list.
     * @param values list
     * @param <T> the type of the values (must implement Comparable)
     * @return a pair: (index of the min value, min value)
     */
    public static <T extends Comparable<T>> Pair<Integer, T> argmin(List<T> values) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("argmin on an empty list");
        }
        //
        // There is no "globally max" value like Inf for an arbitrary type T so we just pick the first list element
        T vmin = values.get(0);
        int imin = 0;
        for (int i = 1; i < values.size(); i++) {
            T v = values.get(i);
            if (v.compareTo(vmin) < 0) {
                vmin = v;
                imin = i;
            }
        }
        return new Pair<>(imin, vmin);
    }

    /**
     * Convert an array of doubles to an array of floats.
     *
     * @param doubles The array of doubles to convert.
     * @return An array of floats.
     */
    public static float[] toFloatArray(double[] doubles) {
        float[] floats = new float[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            floats[i] = (float) doubles[i];
        }
        return floats;
    }

    /**
     * Convert an array of floats to an array of doubles.
     *
     * @param floats The array of floats to convert.
     * @return An array of doubles.
     */
    public static double[] toDoubleArray(float[] floats) {
        double[] doubles = new double[floats.length];
        for (int i = 0; i < floats.length; i++) {
            doubles[i] = floats[i];
        }
        return doubles;
    }

    /**
     * Shuffles the indices in the range [0,size).
     * @param size The number of elements.
     * @param rng The random number generator to use.
     * @return A random permutation of the values in the range (0, size-1).
     */
    public static int[] randperm(int size, Random rng) {
        int[] array = new int[size];
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            int tmp = array[i-1];
            array[i-1] = array[j];
            array[j] = tmp;
        }
        return array;
    }

    /**
     * Shuffles the indices in the range [0,size).
     * @param size The number of elements.
     * @param rng The random number generator to use.
     * @return A random permutation of the values in the range (0, size-1).
     */
    public static int[] randperm(int size, SplittableRandom rng) {
        int[] array = new int[size];
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }
        // Shuffle array
        for (int i = size; i > 1; i--) {
            int j = rng.nextInt(i);
            int tmp = array[i-1];
            array[i-1] = array[j];
            array[j] = tmp;
        }
        return array;
    }

    /**
     * Shuffles the input.
     * @param input The array to shuffle.
     * @param rng The random number generator to use.
     */
    public static void randpermInPlace(int[] input, Random rng) {
        // Shuffle array
        for (int i = input.length; i > 1; i--) {
            int j = rng.nextInt(i);
            int tmp = input[i-1];
            input[i-1] = input[j];
            input[j] = tmp;
        }
    }

    /**
     * Shuffles the input.
     * @param input The array to shuffle.
     * @param rng The random number generator to use.
     */
    public static void randpermInPlace(int[] input, SplittableRandom rng) {
        // Shuffle array
        for (int i = input.length; i > 1; i--) {
            int j = rng.nextInt(i);
            int tmp = input[i-1];
            input[i-1] = input[j];
            input[j] = tmp;
        }
    }

    /**
     * Shuffles the input.
     * @param input The array to shuffle.
     * @param rng The random number generator to use.
     */
    public static void randpermInPlace(double[] input, SplittableRandom rng) {
        // Shuffle array
        for (int i = input.length; i > 1; i--) {
            int j = rng.nextInt(i);
            double tmp = input[i-1];
            input[i-1] = input[j];
            input[j] = tmp;
        }
    }

    /**
     * Draws a bootstrap sample of indices.
     * @param size Size of the sample to generate.
     * @param rng The RNG to use.
     * @return A bootstrap sample.
     */
    public static int[] generateBootstrapIndices(int size, Random rng) {
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = rng.nextInt(size);
        }
        return array;
    }

    /**
     * Draws a bootstrap sample of indices.
     * @param size Size of the sample to generate.
     * @param rng The RNG to use.
     * @return A bootstrap sample.
     */
    public static int[] generateBootstrapIndices(int size, SplittableRandom rng) {
        int[] array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = rng.nextInt(size);
        }
        return array;
    }

    /**
     * Generates a sample of indices weighted by the provided weights.
     * @param size Size of the sample to generate.
     * @param weights A probability mass function of weights.
     * @param rng The RNG to use.
     * @return A sample with replacement from weights.
     */
    public static int[] generateWeightedIndicesSample(int size, double[] weights, Random rng) {
        double[] cdf = generateCDF(weights);
        if (Math.abs(cdf[cdf.length-1] - 1.0) > 1e-10) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length-1]);
        }
        return generateWeightedIndicesSample(cdf, size, rng);
    }

    /**
     * Generates a sample of indices weighted by the provided weights.
     * @param size Size of the sample to generate.
     * @param weights A probability mass function of weights.
     * @param rng The RNG to use.
     * @return A sample with replacement from weights.
     */
    public static int[] generateWeightedIndicesSample(int size, float[] weights, Random rng) {
        double[] cdf = generateCDF(weights);
        if (Math.abs(cdf[cdf.length - 1] - 1.0) > 1e-6) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length - 1]);
        }
        return generateWeightedIndicesSample(cdf, size, rng);
    }

    private static int[] generateWeightedIndicesSample(double[] cdf, int size, Random rng) {
        int[] output = new int[size];

        for (int i = 0; i < output.length; i++) {
            double uniform = rng.nextDouble();
            int searchVal = Arrays.binarySearch(cdf, uniform);
            if (searchVal < 0) {
                output[i] = - 1 - searchVal;
            } else {
                output[i] = searchVal;
            }
        }
        return output;
    }

    /**
     * Generates a sample of indices weighted by the provided weights.
     * @param size Size of the sample to generate.
     * @param weights A probability mass function of weights.
     * @param rng The RNG to use.
     * @return A sample with replacement from weights.
     */
    public static int[] generateWeightedIndicesSample(int size, double[] weights, SplittableRandom rng) {
        double[] cdf = generateCDF(weights);
        if (Math.abs(cdf[cdf.length-1] - 1.0) > 1e-10) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length-1]);
        }
        return generateWeightedIndicesSample(cdf, size, rng);
    }

    /**
     * Generates a sample of indices weighted by the provided weights.
     * @param size Size of the sample to generate.
     * @param weights A probability mass function of weights.
     * @param rng The RNG to use.
     * @return A sample with replacement from weights.
     */
    public static int[] generateWeightedIndicesSample(int size, float[] weights, SplittableRandom rng) {
        double[] cdf = generateCDF(weights);
        if (Math.abs(cdf[cdf.length - 1] - 1.0) > 1e-6) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length - 1]);
        }
        return generateWeightedIndicesSample(cdf, size, rng);
    }

    private static int[] generateWeightedIndicesSample(double[] cdf, int size, SplittableRandom rng) {
        int[] output = new int[size];

        for (int i = 0; i < output.length; i++) {
            double uniform = rng.nextDouble();
            int searchVal = Arrays.binarySearch(cdf, uniform);
            if (searchVal < 0) {
                output[i] = - 1 - searchVal;
            } else {
                output[i] = searchVal;
            }
        }
        return output;
    }

    /**
     * Generates a sample of indices weighted by the provided weights without replacement. Does not recalculate
     * proportions in-between samples. Use judiciously.
     * @param size Size of the sample to generate
     * @param weights A probability mass function of weights
     * @param rng The RNG to use
     * @return A sample without replacement from weights
     */
    public static int[] generateWeightedIndicesSampleWithoutReplacement(int size, double[] weights, Random rng) {
        double[] cdf = generateCDF(weights);
        if (Math.abs(cdf[cdf.length-1] - 1.0) > 1e-6) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length-1]);
        }
        int[] output = new int[size];
        Set<Integer> seenIdxs = new HashSet<>();
        int i = 0;
        while(i < output.length) {
            double uniform = rng.nextDouble();
            int searchVal = Arrays.binarySearch(cdf, uniform);
            int candidateSample = searchVal < 0 ? - 1 - searchVal : searchVal;
            if(!seenIdxs.contains(candidateSample)) {
                seenIdxs.add(candidateSample);
                output[i] = candidateSample;
                i++;
            }
        }
        return output;
    }

    /**
     * Generates a sample of indices weighted by the provided weights without replacement. Does not recalculate
     * proportions in-between samples. Use judiciously.
     * @param size Size of the sample to generate
     * @param weights A probability mass function of weights
     * @param rng The RNG to use
     * @return A sample without replacement from weights
     */
    public static int[] generateWeightedIndicesSampleWithoutReplacement(int size, float[] weights, Random rng) {
        double[] cdf = generateCDF(weights);
        if (Math.abs(cdf[cdf.length-1] - 1.0) > 1e-6) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length-1]);
        }
        int[] output = new int[size];
        Set<Integer> seenIdxs = new HashSet<>();
        int i = 0;
        while(i < output.length) {
            double uniform = rng.nextDouble();
            int searchVal = Arrays.binarySearch(cdf, uniform);
            int candidateSample = searchVal < 0 ? - 1 - searchVal : searchVal;
            if(!seenIdxs.contains(candidateSample)) {
                seenIdxs.add(candidateSample);
                output[i] = candidateSample;
                i++;
            }
        }
        return output;
    }

    /**
     * Generates a cumulative distribution function from the supplied probability mass function.
     * @param pmf The probability mass function (i.e., the probability distribution).
     * @return The CDF.
     */
    public static double[] generateCDF(double[] pmf) {
        return cumulativeSum(pmf);
    }

    /**
     * Produces a cumulative sum array.
     * @param input The input to sum.
     * @return The cumulative sum.
     */
    public static double[] cumulativeSum(double[] input) {
        double[] cdf = new double[input.length];

        double sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += input[i];
            cdf[i] = sum;
        }

        return cdf;
    }

    /**
     * Produces a cumulative sum array.
     * @param input The input to sum.
     * @return The cumulative sum.
     */
    public static int[] cumulativeSum(boolean[] input) {
        int[] cumulativeSum = new int[input.length];

        int sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] ? 1 : 0;
            cumulativeSum[i] = sum;
        }

        return cumulativeSum;
    }

    /**
     * Generates a cumulative distribution function from the supplied probability mass function.
     * @param pmf The probability mass function (i.e., the probability distribution).
     * @return The CDF.
     */
    public static double[] generateCDF(float[] pmf) {
        double[] cdf = new double[pmf.length];

        double sum = 0;
        for (int i = 0; i < pmf.length; i++) {
            sum += pmf[i];
            cdf[i] = sum;
        }
        
        return cdf;
    }

    /**
     * Generates a cumulative distribution function from the supplied probability mass function.
     * @param counts The frequency counts.
     * @param countSum The sum of the counts.
     * @return The CDF.
     */
    public static double[] generateCDF(long[] counts, long countSum) {
        double[] cdf = new double[counts.length];

        double countSumD = countSum;
        double probSum = 0.0;
        for (int i = 0; i < counts.length; i++) {
            probSum += counts[i] / countSumD;
            cdf[i] = probSum;
        }

        return cdf;
    }

    /**
     * Samples an index from the supplied cdf.
     * @param cdf The cdf to sample from.
     * @param rng The rng to use.
     * @return A sample.
     */
    public static int sampleFromCDF(double[] cdf, Random rng) {
        if (Math.abs(cdf[cdf.length-1] - 1.0) > 1e-6) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length-1]);
        }
        double uniform = rng.nextDouble();
        int searchVal = Arrays.binarySearch(cdf, uniform);
        if (searchVal < 0) {
            return - 1 - searchVal;
        } else {
            return searchVal;
        }
    }

    /**
     * Samples an index from the supplied cdf.
     * @param cdf The cdf to sample from.
     * @param rng The rng to use.
     * @return A sample.
     */
    public static int sampleFromCDF(double[] cdf, SplittableRandom rng) {
        if (Math.abs(cdf[cdf.length-1] - 1.0) > 1e-6) {
            throw new IllegalStateException("Weights do not sum to 1, cdf[cdf.length-1] = " + cdf[cdf.length-1]);
        }
        double uniform = rng.nextDouble();
        int searchVal = Arrays.binarySearch(cdf, uniform);
        if (searchVal < 0) {
            return - 1 - searchVal;
        } else {
            return searchVal;
        }
    }

    public static double[] generateUniformVector(int length, double value) {
        double[] output = new double[length];

        Arrays.fill(output, value);

        return output;
    }

    public static float[] generateUniformVector(int length, float value) {
        float[] output = new float[length];

        Arrays.fill(output, value);

        return output;
    }

    public static double[] normalizeToDistribution(double[] input) {
        double[] output = new double[input.length];
        double sum = 0.0;

        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
            sum += output[i];
        }

        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    public static double[] normalizeToDistribution(float[] input) {
        double[] output = new double[input.length];
        double sum = 0.0;

        for (int i = 0; i < input.length; i++) {
            output[i] = input[i];
            sum += output[i];
        }

        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        return output;
    }

    public static double[] inplaceNormalizeToDistribution(double[] input) {
        double sum = 0.0;

        for (int i = 0; i < input.length; i++) {
            sum += input[i];
        }

        for (int i = 0; i < input.length; i++) {
            input[i] /= sum;
        }

        return input;
    }

    public static void inplaceNormalizeToDistribution(float[] input) {
        float sum = 0.0f;

        for (int i = 0; i < input.length; i++) {
            sum += input[i];
        }

        for (int i = 0; i < input.length; i++) {
            input[i] /= sum;
        }

    }

    public static void logVector(Logger otherLogger, Level level, double[] input) {
        StringBuilder buffer = new StringBuilder();

        for (int i = 0; i < input.length; i++) {
            buffer.append("(");
            buffer.append(i);
            buffer.append(",");
            buffer.append(input[i]);
            buffer.append(") ");
        }
        buffer.deleteCharAt(buffer.length()-1);
        otherLogger.log(level, buffer.toString());
    }

    public static void logVector(Logger otherLogger, Level level, float[] input) {
        StringBuilder buffer = new StringBuilder();

        for (int i = 0; i < input.length; i++) {
            buffer.append("(");
            buffer.append(i);
            buffer.append(",");
            buffer.append(input[i]);
            buffer.append(") ");
        }
        buffer.deleteCharAt(buffer.length()-1);
        otherLogger.log(level, buffer.toString());
    }

    public static double[] toPrimitiveDoubleFromInteger(List<Integer> input) {
        double[] output = new double[input.size()];

        for (int i = 0; i < input.size(); i++) {
            output[i] = input.get(i);
        }

        return output;
    }

    public static double[] toPrimitiveDouble(List<Double> input) {
        double[] output = new double[input.size()];

        for (int i = 0; i < input.size(); i++) {
            output[i] = input.get(i);
        }
        
        return output;
    }

    public static float[] toPrimitiveFloat(List<Float> input) {
        float[] output = new float[input.size()];

        for (int i = 0; i < input.size(); i++) {
            output[i] = input.get(i);
        }
        
        return output;
    }

    public static int[] toPrimitiveInt(List<Integer> input) {
        int[] output = new int[input.size()];

        for (int i = 0; i < input.size(); i++) {
            output[i] = input.get(i);
        }
        
        return output;
    }

    public static long[] toPrimitiveLong(List<Long> input) {
        long[] output = new long[input.size()];

        for (int i = 0; i < input.size(); i++) {
            output[i] = input.get(i);
        }
        
        return output;
    }

    public static int[] sampleInts(Random rng, int size, int range) {
        int[] output = new int[size];

        for (int i = 0; i < output.length; i++) {
            output[i] = rng.nextInt(range);
        }

        return output;
    }

    public static void inPlaceAdd(double[] input, double[] update) {
        for (int i = 0; i < input.length; i++) {
            input[i] += update[i];
        }
    }

    public static void inPlaceSubtract(double[] input, double[] update) {
        for (int i = 0; i < input.length; i++) {
            input[i] -= update[i];
        }
    }

    public static void inPlaceAdd(float[] input, float[] update) {
        for (int i = 0; i < input.length; i++) {
            input[i] += update[i];
        }
    }

    public static void inPlaceSubtract(float[] input, float[] update) {
        for (int i = 0; i < input.length; i++) {
            input[i] -= update[i];
        }
    }

    public static double vectorNorm(double[] input) {
        double norm = 0.0;
        for (double d : input) {
            norm += d * d;
        }
        return norm;
    }

    public static double sum(double[] input) {
        double sum = 0.0;
        for (double d : input) {
            sum += d;
        }
        return sum;
    }

    public static float sum(float[] input) {
        float sum = 0.0f;
        for (float d : input) {
            sum += d;
        }
        return sum;
    }

    public static double sum(double[] array, int length) {
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += array[i];
        }
        return sum;
    }

    public static float sum(float[] array, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += array[i];
        }
        return sum;
    }

    public static float sum(int[] indices, int indicesLength, float[] input) {
        float sum = 0.0f;
        for (int i = 0; i < indicesLength; i++) {
            sum += input[indices[i]];
        }
        return sum;
    }

    public static float sum(int[] indices, float[] input) {
        return sum(indices,indices.length,input);
    }

    public static float[] generateUniformFloatVector(int length, float value) {
        float[] output = new float[length];

        Arrays.fill(output, value);

        return output;
    }

    /**
     * A binary search function.
     * @param list Input list, must be ordered.
     * @param key Key to search for.
     * @param <T> Type of the list, must implement Comparable.
     * @return the index of the search key, if it is contained in the list;
     *         otherwise, (-(insertion point) - 1). The insertion point is
     *         defined as the point at which the key would be inserted into
     *         the list: the index of the first element greater than the key,
     *         or list.size() if all elements in the list are less than the
     *         specified key. Note that this guarantees that the return value
     *         will be &gt;= 0 if and only if the key is found.
     */
    public static <T> int binarySearch(List<? extends Comparable<? super T>> list, T key) {
        return binarySearch(list,key,0,list.size()-1);
    }

    /**
     * A binary search function.
     * @param list Input list, must be ordered.
     * @param key Key to search for.
     * @param low Starting index.
     * @param high End index (will be searched).
     * @param <T> Type of the list, must implement Comparable.
     * @return the index of the search key, if it is contained in the list;
     *         otherwise, (-(insertion point) - 1). The insertion point is
     *         defined as the point at which the key would be inserted into
     *         the list: the index of the first element greater than the key,
     *         or high if all elements in the list are less than the
     *         specified key. Note that this guarantees that the return value
     *         will be &gt;= 0 if and only if the key is found.
     */
    public static <T> int binarySearch(List<? extends Comparable<? super T>> list, T key, int low, int high) {
        while (low <= high) {
            int mid = (low + high) >>> 1;
            Comparable<? super T> midVal = list.get(mid);
            int cmp = midVal.compareTo(key);
            if (cmp < 0) {
                low = mid + 1;
            } else if (cmp > 0) {
                high = mid - 1;
            } else {
                return mid; // key found
            }
        }
        return -(low + 1);  // key not found
    }

    /**
     * A binary search function.
     * @param list Input list, must be ordered.
     * @param key Key to search for.
     * @param extractionFunc Takes a T and generates an int
     *                       which can be used for comparison using int's natural ordering.
     * @param <T> Type of the list, must implement Comparable.
     * @return the index of the search key, if it is contained in the list;
     *         otherwise, (-(insertion point) - 1). The insertion point is
     *         defined as the point at which the key would be inserted into
     *         the list: the index of the first element greater than the key,
     *         or high if all elements in the list are less than the
     *         specified key. Note that this guarantees that the return value
     *         will be &gt;= 0 if and only if the key is found.
     */
    public static <T> int binarySearch(List<? extends T> list, int key, ToIntFunction<T> extractionFunc) {
        int low = 0;
        int high = list.size()-1;
        while (low <= high) {
            int mid = (low + high) >>> 1;
            int midVal = extractionFunc.applyAsInt(list.get(mid));
            int cmp = Integer.compare(midVal, key);
            if (cmp < 0) {
                low = mid + 1;
            } else if (cmp > 0) {
                high = mid - 1;
            } else {
                return mid; // key found
            }
        }
        return -(low + 1);  // key not found
    }

    /**
     * Calculates the area under the curve, bounded below by the x axis.
     * <p>
     * Uses linear interpolation between the points on the x axis,
     * i.e., trapezoidal integration.
     * <p>
     * The x axis must be increasing.
     * @param x The x points to evaluate.
     * @param y The corresponding heights.
     * @return The AUC.
     */
    public static double auc(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new IllegalArgumentException("x and y must be the same length, x.length = " + x.length + ", y.length = " + y.length);
        }
        double output = 0.0;

        for (int i = 1; i < x.length; i++) {
            double ySum = y[i] + y[i-1];
            double xDiff = x[i] - x[i-1];
            if (xDiff < -1e-12) {
                throw new IllegalStateException(String.format("X is not increasing, x[%d]=%f, x[%d]=%f",i,x[i],i-1,x[i-1]));
            }
            output += (ySum * xDiff) / 2.0;
        }

        return output;
    }

    /**
     * Returns the mean and variance of the input.
     * @param inputs The input array.
     * @return The mean and variance of the inputs. The mean is the first element, the variance is the second.
     */
    public static Pair<Double,Double> meanAndVariance(double[] inputs) {
        return meanAndVariance(inputs,inputs.length);
    }

    /**
     * Returns the mean and variance of the input's first length elements.
     * @param inputs The input array.
     * @param length The number of elements to use.
     * @return The mean and variance of the inputs. The mean is the first element, the variance is the second.
     */
    public static Pair<Double,Double> meanAndVariance(double[] inputs, int length) {
        double mean = 0.0;
        double sumSquares = 0.0;
        for (int i = 0; i < length; i++) {
            double value = inputs[i];
            double delta = value - mean;
            mean += delta / (i+1);
            double delta2 = value - mean;
            sumSquares += delta * delta2;
        }
        return new Pair<>(mean,sumSquares/(length-1));
    }

    /**
     * Returns the weighted mean of the input.
     * <p>
     * Throws IllegalArgumentException if the two arrays are not the same length.
     * @param inputs The input array.
     * @param weights The weights to use.
     * @return The weighted mean.
     */
    public static double weightedMean(double[] inputs, double[] weights) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("inputs and weights must be the same length, inputs.length = " + inputs.length + ", weights.length = " + weights.length);
        }

        double output = 0.0;
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            output += inputs[i] * weights[i];
            sum += weights[i];
        }

        return output/sum;
    }

    /**
     * Returns the mean of the input array.
     * @param inputs The input array.
     * @return The mean of inputs.
     */
    public static double mean(double[] inputs) {
        double output = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            output += inputs[i];
        }
        return output / inputs.length;
    }

    public static double mean(double[] array, int length) {
        double sum = sum(array,length);
        return sum / length;
    }

    public static <V extends Number> double mean(Collection<V> values) {
        double total = 0d;
        for (V v : values) {
            total += v.doubleValue();
        }
        return total / values.size();
    }

    public static <V extends Number> double sampleVariance(Collection<V> values) {
        double mean = mean(values);
        double total = 0d;
        for (V v : values) {
            total += Math.pow(v.doubleValue()-mean, 2);
        }
        return total / (values.size() - 1);
    }

    public static <V extends Number> double sampleStandardDeviation(Collection<V> values) {
        return Math.sqrt(sampleVariance(values));
    }

    public static double weightedMean(double[] array, float[] weights, int length) {
        double sum = weightedSum(array,weights,length);
        return sum / sum(weights,length);
    }

    public static double weightedSum(double[] array, float[] weights, int length) {
        if (array.length != weights.length) {
            throw new IllegalArgumentException("array and weights must be the same length, array.length = " + array.length + ", weights.length = " + weights.length);
        }

        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            sum += weights[i] * array[i];
        }
        return sum;
    }

    /**
     * Returns an array containing the indices where values are different.
     * Basically a combination of np.where and np.diff.
     * <p>
     * Stores an index if the value after it is different. Always stores the
     * final index.
     * <p>
     * Uses a default tolerance of 1e-12.
     * @param input Input array.
     * @return An array containing the indices where the input changes.
     */
    public static int[] differencesIndices(double[] input) {
        return differencesIndices(input,1e-12);
    }

    /**
     * Returns an array containing the indices where values are different.
     * Basically a combination of np.where and np.diff.
     * <p>
     * Stores an index if the value after it is different. Always stores the
     * final index.
     * @param input Input array.
     * @param tolerance Tolerance to determine a difference.
     * @return An array containing the indices where the input changes.
     */
    public static int[] differencesIndices(double[] input, double tolerance) {
        List<Integer> indices = new ArrayList<>();

        for (int i = 0; i < input.length-1; i++) {
            double diff = Math.abs(input[i+1] - input[i]);
            if (diff > tolerance) {
                indices.add(i);
            }
        }
        indices.add(input.length-1);

        return Util.toPrimitiveInt(indices);
    }

    /**
     * Formats a duration given two times in milliseconds.
     * <p>
     * Format string is - (%02d:%02d:%02d:%03d) or (%d days, %02d:%02d:%02d:%03d)
     *
     * @param startMillis Start time in ms.
     * @param stopMillis End time in ms.
     * @return A formatted string measuring time in hours, minutes, second and milliseconds.
     */
    public static String formatDuration(long startMillis, long stopMillis) {
        long millis = stopMillis - startMillis;
        long second = (millis / 1000) % 60;
        long minute = (millis / (1000 * 60)) % 60;
        long hour = (millis / (1000 * 60 * 60)) % 24;
        long days = (millis / (1000 * 60 * 60)) / 24;

        if (days == 0) {
            return String.format("(%02d:%02d:%02d:%03d)", hour, minute, second, millis % 1000);
        } else {
            return String.format("(%d days, %02d:%02d:%02d:%03d)", days, hour, minute, second, millis % 1000);
        }
    }

    /**
     * Expects sorted input arrays. Returns an array containing all the elements in first that are not in second.
     * @param first The first sorted array.
     * @param second The second sorted array.
     * @return An array containing all the elements of first that aren't in second.
     */
    public static int[] sortedDifference(int[] first, int[] second) {
        List<Integer> diffIndicesList = new ArrayList<>();

        int i = 0;
        int j = 0;
        while (i < first.length && j < second.length) {
            //after this loop, either itr is out or tuple.index >= otherTuple.index
            while (i < first.length && (first[i] < second[j])) {
                diffIndicesList.add(first[i]);
                i++;
            }
            //after this loop, either otherItr is out or tuple.index <= otherTuple.index
            while (j < second.length && (first[i] > second[j])) {
                j++;
            }
            if (first[i] != second[j]) {
                diffIndicesList.add(first[i]);
            }
        }
        for (; i < first.length; i++) {
            diffIndicesList.add(first[i]);
        }
        return diffIndicesList.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Standardizes the input, i.e. divides it by the variance and subtracts the mean.
     * @param input The input to standardize.
     * @param mean The mean.
     * @param variance The variance.
     * @return The standardized input.
     */
    public static double[] standardize(double[] input, double mean, double variance) {
        if (variance <= 0.0) {
            throw new IllegalArgumentException("Variance must be positive, found " + variance);
        }
        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (input[i] - mean) / variance;
        }
        return output;
    }

    /**
     * Standardizes the input, i.e. divides it by the variance and subtracts the mean.
     * @param input The input to standardize.
     * @param mean The mean.
     * @param variance The variance.
     */
    public static void standardizeInPlace(double[] input, double mean, double variance) {
        if (variance <= 0.0) {
            throw new IllegalArgumentException("Variance must be positive, found " + variance);
        }
        for (int i = 0; i < input.length; i++) {
            input[i] = (input[i] - mean) / variance;
        }
    }
}
