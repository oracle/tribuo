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

package org.tribuo.common.tree.impl;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * An array container which maintains the array and the size.
 * This class is barely more than a Tuple, it's up to you to maintain the size invariant.
 */
public class IntArrayContainer {
    private static final Logger logger = Logger.getLogger(IntArrayContainer.class.getName());

    /**
     * The array of ints.
     */
    public int[] array;
    /**
     * The number of elements in the array.
     */
    public int size;

    /**
     * Constructs a new int array container with the specified initial backing array size.
     * @param initialCapacity The initial capacity of the backing array.
     */
    public IntArrayContainer(int initialCapacity) {
        array = new int[initialCapacity];
        size = 0;
    }

    /**
     * Grows the backing array, copying the elements.
     * @param requestedSize The size to grow the array to.
     */
    public void grow(int requestedSize) {
        if (requestedSize > array.length) {
            // overflow-conscious code
            int oldCapacity = array.length;
            int newCapacity = oldCapacity + (oldCapacity >> 1);
            if (newCapacity - requestedSize < 0) {
                newCapacity = requestedSize;
            }
            // minCapacity is usually close to size, so this is a win:
            array = Arrays.copyOf(array, newCapacity);
        }
    }

    /**
     * Returns a copy of the elements in use.
     * @return A copy of the elements.
     */
    public int[] copy() {
        return Arrays.copyOf(array,size);
    }

    /**
     * Overwrites values from the supplied array into this array.
     * @param otherArray The array to copy from.
     */
    public void fill(int[] otherArray) {
        if (otherArray.length > array.length) {
            array = Arrays.copyOf(otherArray,otherArray.length);
        } else {
            System.arraycopy(otherArray,0,array,0,otherArray.length);
        }
        size = otherArray.length;
    }

    /**
     * Overwrites values in this array with the supplied array.
     * @param other The array to copy from.
     */
    public void fill(IntArrayContainer other) {
        if (other.array.length > array.length) {
            array = Arrays.copyOf(other.array,other.size);
        } else {
            System.arraycopy(other.array,0,array,0,other.size);
        }
        size = other.size;
    }

    /**
     * Copies from input to output excluding the values in otherArray.
     * <p>
     * This assumes both input and otherArray are sorted. Behaviour is undefined if they aren't.
     * @param input The input container.
     * @param otherArray Another (sorted) int array.
     * @param output The container to write the output to.
     */
    public static void removeOther(IntArrayContainer input, int[] otherArray, IntArrayContainer output) {
        //logger.info("input.size = " + input.size + ", otherArray.length = " + otherArray.length + ", output.length = " + output.array.length);
        int newSize = input.size - otherArray.length;
        if (newSize > output.array.length) {
            output.grow(newSize);
        }

        int[] inputArray = input.array;
        int inputSize = input.size;
        int[] outputArray = output.array;
        //logger.info("input = " + Arrays.toString(inputArray));
        //logger.info("otherArray = " + Arrays.toString(otherArray));

        int i = 0; //index into input
        int j = 0; //index into otherArray
        int k = 0; //index into output
        while (i < inputSize) {
            if (j == otherArray.length) {
                // Reached end of other, copy from input
                outputArray[k] = inputArray[i];
                i++;
                k++;
            } else if (inputArray[i] < otherArray[j]) {
                // Input less than other, copy input
                outputArray[k] = inputArray[i];
                i++;
                k++;
            } else if (inputArray[i] == otherArray[j]) {
                // skip both
                i++;
                j++;
            } else {
                // other less than input, skip
                j++;
            }
        }
        output.size = k;
        assert(k == newSize);
        //logger.info("output = " + Arrays.toString(outputArray));
    }

    /**
     * Merges the list of int arrays into a single int array, using the two supplied buffers.
     * Requires that all arrays in the list are sorted, and that they contain unique values.
     * @param input A list of int arrays.
     * @param firstBuffer A buffer.
     * @param secondBuffer Another buffer.
     * @return A sorted array containing all the elements from the input.
     */
    public static int[] merge(List<int[]> input, IntArrayContainer firstBuffer, IntArrayContainer secondBuffer) {
        if (input.size() > 0) {
            firstBuffer.fill(input.get(0));
            for (int i = 0; i < input.size(); i++) {
                merge(firstBuffer,input.get(i),secondBuffer);
                IntArrayContainer tmp = secondBuffer;
                secondBuffer = firstBuffer;
                firstBuffer = tmp;
            }
            return firstBuffer.copy();
        } else {
            return new int[0];
        }
    }

    /**
     * Merges input and otherArray writing to output.
     * This assumes both input and otherArray are sorted. Behaviour is undefined if they aren't.
     * @param input The input container.
     * @param otherArray Another (sorted) int array.
     * @param output The container to write the output to.
     */
    public static void merge(IntArrayContainer input, int[] otherArray, IntArrayContainer output) {
        int newSize = input.size + otherArray.length;
        if (newSize > output.array.length) {
            output.grow(newSize);
        }

        int[] inputArray = input.array;
        int inputSize = input.size;
        int[] outputArray = output.array;

        int i = 0; //index into input
        int j = 0; //index into otherArray
        int k = 0; //index into output
        while ((i < inputSize) || (j < otherArray.length)) {
            if (i == inputSize) {
                // Reached end of input, copy from other
                outputArray[k] = otherArray[j];
                j++;
                k++;
            } else if (j == otherArray.length) {
                // Reached end of other, copy from input
                outputArray[k] = inputArray[i];
                i++;
                k++;
            } else if (inputArray[i] < otherArray[j]) {
                // Input less than other, copy input
                outputArray[k] = inputArray[i];
                i++;
                k++;
            } else {
                // other less than input, copy other
                outputArray[k] = otherArray[j];
                j++;
                k++;
            }
        }
        output.size = k;
        assert(k == newSize);
        //logger.info("input = " + Arrays.toString(inputArray));
        //logger.info("otherArray = " + Arrays.toString(otherArray));
        //logger.info("output = " + Arrays.toString(outputArray));
    }
}
