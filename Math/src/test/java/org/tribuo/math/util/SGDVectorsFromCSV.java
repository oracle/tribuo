/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math.util;

import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * A Test helper class which provides a mechanism to obtain an array of SGDVectors from a CVS file.
 */
public class SGDVectorsFromCSV {

    public static SGDVector[] getSGDVectorsFromCSV(URL filePath, boolean fileContainsHeader) {
        List<SGDVector> vectorList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(filePath.openStream()))) {
            String line;

            if (fileContainsHeader) {
                line = br.readLine();
                System.out.println("The header line: " + line + " was ignored.");
            }

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] doubles = new double[values.length];
                for (int i=0; i<values.length; i++) {
                    doubles[i] = Double.parseDouble(values[i]);
                }
                SGDVector vector = DenseVector.createDenseVector(doubles);
                vectorList.add(vector);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("\nThere is a problem with the file " + filePath);
            throw new UncheckedIOException(e);
        }
        return vectorList.toArray(new SGDVector[0]);
    }

}
