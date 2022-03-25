/*
 * Copyright (c) 2015-2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.data.csv;

import com.opencsv.CSVParserWriter;
import com.opencsv.ICSVWriter;
import com.opencsv.RFC4180ParserBuilder;
import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import org.tribuo.Dataset;
import org.tribuo.Example;
import org.tribuo.Feature;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Output;
import org.tribuo.VariableIDInfo;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Saves a Dataset in CSV format suitable for loading by {@link CSVLoader}.
 * <p>
 * CSVSaver is thread safe and immutable.
 */
public class CSVSaver implements Configurable {

    private static final Logger logger = Logger.getLogger(CSVSaver.class.getName());

    /**
     * The default response column name.
     */
    public final static String DEFAULT_RESPONSE = "Response";

    @Config(description="The column separator.")
    private char separator = CSVIterator.SEPARATOR;
    @Config(description="The quote character.")
    private char quote = CSVIterator.QUOTE;

    /**
     * Builds a CSV saver using the supplied separator and quote.
     * @param separator The column separator.
     * @param quote The quote character.
     */
    public CSVSaver(char separator, char quote) {
        if (separator == quote) {
            throw new IllegalArgumentException("Quote and separator must be different characters.");
        }
        this.separator = separator;
        this.quote = quote;
    }

    /**
     * Builds a CSV saver using the default separator and quote from {@link CSVIterator}.
     */
    public CSVSaver() {
        this(CSVIterator.SEPARATOR, CSVIterator.QUOTE);
    }

    /**
     * Saves the dataset to the specified path.
     * @param csvPath The path to save to.
     * @param dataset The dataset to save.
     * @param responseName The name of the response variable.
     * @param <T> The output type.
     * @throws IOException If the disk write failed.
     */
    public <T extends Output<T>> void save(Path csvPath, Dataset<T> dataset, String responseName) throws IOException {
        save(csvPath, dataset, Collections.singleton(responseName));
    }

    /**
     * Saves the dataset to the specified path.
     * @param csvPath The path to save to.
     * @param dataset The dataset to save.
     * @param responseNames The response names set.
     * @param <T> The output type.
     * @throws IOException If the disk write failed.
     */
    public <T extends Output<T>> void save(Path csvPath, Dataset<T> dataset, Set<String> responseNames) throws IOException {
        boolean isMultiOutput = responseNames.size() > 1;
        ImmutableFeatureMap features = dataset.getFeatureIDMap();
        int ncols = features.size() + responseNames.size();
        //
        // Initialize the CSV header row.
        String[] headerLine = new String[ncols];
        Map<String, Integer> responseToColumn = new HashMap<>();
        int col = 0;
        for (String response : responseNames) {
            headerLine[col] = response;
            responseToColumn.put(response, col);
            col++;
        }
        for (int i = 0; i < features.size(); i++) {
            headerLine[col++] = features.get(i).getName();
        }
        //
        // Write the CSV
        try (ICSVWriter writer = new CSVParserWriter(
                Files.newBufferedWriter(csvPath, StandardCharsets.UTF_8),
                new RFC4180ParserBuilder()
                        .withSeparator(separator)
                        .withQuoteChar(quote)
                        .build(), "\n")) {

            writer.writeNext(headerLine);

            for (Example<T> e : dataset) {
                String[] denseOutput = (isMultiOutput) ?
                        densifyMultiOutput(e, responseToColumn) :
                        densifySingleOutput(e);
                String[] featureArr = generateFeatureArray(e, features);
                if (featureArr.length != features.size()) {
                    throw new IllegalStateException(String.format("Invalid example: had %d features, expected %d.", featureArr.length, features.size()));
                }
                //
                // Copy responses and features into a single array
                String[] line = new String[ncols];
                System.arraycopy(denseOutput, 0, line, 0, denseOutput.length);
                System.arraycopy(featureArr, 0, line, denseOutput.length, featureArr.length);
                writer.writeNext(line);
            }
        }
    }

    private static <T extends Output<T>> String[] densifySingleOutput(Example<T> e) {
        return new String[]{e.getOutput().getSerializableForm(false)};
    }

    private static <T extends Output<T>> String[] densifyMultiOutput(Example<T> e, Map<String, Integer> responseToColumn) {
        String[] denseOutput = new String[responseToColumn.size()];
        //
        // Initialize to false/0 everywhere
        // TODO ^^^ seems bad to me. maybe OutputFactory could give us a "zero-value" in addition to an "unknown-value".
        //TODO: maybe this instead? outputFactory.getUnknownOutput().toString();
        Arrays.fill(denseOutput, "0");

        //
        // Convert sparse output to a dense format
        // Sparse output format: "a=true,b=true..." for classification or "a=0.0,b=2.0..." for regression
        String csv = e.getOutput().getSerializableForm(false);
        if (csv.isEmpty()) {
            //
            // If the string is empty, then the denseOutput will be false/0 everywhere.
            return denseOutput;
        }
        String[] sparseOutput = csv.split(","); // TODO should comma be hard-coded into this 'split' call?

        for (String elem : sparseOutput) {
            String[] kv = elem.split("=");
            if (kv.length != 2) {
                throw new IllegalArgumentException("Bad serialized string element: '" + elem + "'");
            }
            String responseName = kv[0];
            String responseValue = kv[1];
            int index = responseToColumn.getOrDefault(responseName,-1);
            if (index == -1) {
                //
                // We have to check for a special-case here:
                // In the multi-output case, we might have a CSV like the following:
                //
                // Feature1,Feature2,...,Label1,Label2
                // 1.0,0.5,...,False,False
                //
                // In this case, where we have false for all labels, the multi-output label will just be the
                // empty string. In the single label case, an empty-string label should be an error.
                if (responseName.equals("")) {
                    continue;
                } else {
                    throw new IllegalStateException(String.format("Invalid example: unknown response name '%s'. (known response names: %s)", responseName, responseToColumn.keySet()));
                }
            }
            denseOutput[index] = responseValue;
        }

        return denseOutput;
    }

    /**
     * Converts an Example's features into a dense row, filling in unobserved values with 0.
     * @param example The example to convert.
     * @param features The featureMap to use for the ids.
     * @return A String array, one element per feature plus the output.
     */
    private static <T extends Output<T>> String[] generateFeatureArray(Example<T> example, ImmutableFeatureMap features) {
        String[] output = new String[features.size()];
        HashMap<Integer,Double> featureMap = new HashMap<>();
        for (Feature f : example) {
            VariableIDInfo info = features.get(f.getName());
            if (info != null) {
                featureMap.put(info.getID(), f.getValue());
            }
        }
        for (int i = 0; i < features.size(); i++) {
            Double curFeature = featureMap.get(i);
            if (curFeature == null) {
                output[i] = "0";
            } else {
                output[i] = curFeature.toString();
            }
        }
        return output;
    }

}
