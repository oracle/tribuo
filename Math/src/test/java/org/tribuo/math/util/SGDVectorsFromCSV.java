package org.tribuo.math.util;

import org.junit.jupiter.api.Test;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.SGDVector;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A Test helper class which provides a mechanism to obtain an array of SGDVectors from a CVS file.
 */
public class SGDVectorsFromCSV {

    public static SGDVector[] getSGDVectorsFromCSV(String filePath, boolean fileContainsHeader) {
        List<SGDVector> vectorList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
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
        }
        return vectorList.toArray(new SGDVector[0]);
    }

}
