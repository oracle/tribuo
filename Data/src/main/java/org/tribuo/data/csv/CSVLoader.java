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

package org.tribuo.data.csv;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.CharProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.URLProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.datasource.ListDataSource;
import org.tribuo.impl.ArrayExample;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Load a DataSource/Dataset from a CSV file.
 * @param <T> The type of the output generated.
 */
public class CSVLoader<T extends Output<T>> {

    private static final Logger logger = Logger.getLogger(CSVLoader.class.getName());

    private final char separator;
    private final char quote;
    private final OutputFactory<T> outputFactory;

    public CSVLoader(char separator, char quote, OutputFactory<T> outputFactory) {
        this.separator = separator;
        this.quote = quote;
        this.outputFactory = outputFactory;
    }

    public CSVLoader(char separator, OutputFactory<T> outputFactory) {
        this(separator, CSVIterator.QUOTE,outputFactory);
    }

    public CSVLoader(OutputFactory<T> outputFactory) {
    this(CSVIterator.SEPARATOR,CSVIterator.QUOTE,outputFactory);
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     * @param csvPath The path to load.
     * @param responseName The name of the response variable.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, String responseName) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseName));
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     * @param csvPath The path to load.
     * @param responseName The name of the response variable.
     * @param header The header of the CSV if it's not present in the file.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, String responseName, String[] header) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseName, header));
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     * @param csvPath The path to load.
     * @param responseNames The names of the response variables.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, Set<String> responseNames) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseNames));
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     * @param csvPath The path to load.
     * @param responseNames The names of the response variables.
     * @param header The header of the CSV if it's not present in the file.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, Set<String> responseNames, String[] header) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseNames, header));
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseName The name of the response variable.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(Path csvPath, String responseName) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName));
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseName The name of the response variable.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(URL csvPath, String responseName) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName));
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseName The name of the response variable.
     * @param header The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(Path csvPath, String responseName, String[] header) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName), header);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseName The name of the response variable.
     * @param header The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(URL csvPath, String responseName, String[] header) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName), header);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseNames The names of the response variables.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(Path csvPath, Set<String> responseNames) throws IOException {
        return loadDataSource(csvPath,responseNames,null);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseNames The names of the response variables.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(URL csvPath, Set<String> responseNames) throws IOException {
        return loadDataSource(csvPath,responseNames,null);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseNames The names of the response variables.
     * @param header The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(Path csvPath, Set<String> responseNames, String[] header) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(csvPath)) {
            URL url = csvPath.toUri().toURL();
            return loadDataSource(reader,url,responseNames,header);
        }
    }

    /**
     * Loads a DataSource from the specified csv path.
     * @param csvPath The csv to load from.
     * @param responseNames The names of the response variables.
     * @param header The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public ListDataSource<T> loadDataSource(URL csvPath, Set<String> responseNames, String[] header) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(csvPath.openStream(), StandardCharsets.UTF_8))) {
            return loadDataSource(reader,csvPath,responseNames,header);
        }
    }

    private ListDataSource<T> loadDataSource(Reader reader, URL path, Set<String> responseNames, String[] header) throws IOException {
        try (CSVIterator itr = new CSVIterator(reader, separator, quote, header)) {
            //
            // CSVInteropProvenance constructor throws an exception on FileNotFound, so we include in the try/catch
            DataSourceProvenance provenance = new CSVLoaderProvenance(
                    path,
                    outputFactory.getProvenance(),
                    String.join(",", responseNames), // If there are multiple responses, join them
                    separator,
                    quote
            );
            List<Example<T>> list = innerLoadFromCSV(itr, responseNames, path.toString());
            return new ListDataSource<>(list, outputFactory, provenance);
        }
    }

    private List<Example<T>> innerLoadFromCSV(CSVIterator itr, Set<String> responseNames, String csvPath) {
        validateResponseNames(responseNames, itr.getFields(), csvPath);
        List<Example<T>> dataset = new ArrayList<>();
        String responseName = responseNames.size() == 1 ? responseNames.iterator().next() : null;
        //
        // Create the examples.
        while (itr.hasNext()) {
            Map<String, String> row = itr.next().getRowData();
            T label = (responseNames.size() == 1) ?
                    buildOutput(responseName, row) :
                    buildMultiOutput(responseNames, row);
            ArrayExample<T> example = new ArrayExample<>(label);
            for (Map.Entry<String, String> ent : row.entrySet()) {
                String columnName = ent.getKey();
                if (!responseNames.contains(columnName)) {
                    //
                    // If it's not a response, it's a feature
                    double value = Double.parseDouble(ent.getValue());
                    example.add(columnName, value);
                }
            }
            dataset.add(example);
        }
        return dataset;
    }

    private static void validateResponseNames(Set<String> responseNames, List<String> headers, String csvPath) throws IllegalArgumentException {
        if (responseNames.isEmpty()) {
            throw new IllegalArgumentException("At least one response name must be specified, but responseNames is empty.");
        }
        //
        // Validate that all the expected responses are included in the given header fields
        Map<String, Boolean> responsesFound = new HashMap<>();
        for (String response : responseNames) {
            responsesFound.put(response, false);
        }
        for (String header : headers) {
            if (responseNames.contains(header)) {
                responsesFound.put(header, true);
            }
        }
        for (Map.Entry<String, Boolean> kv : responsesFound.entrySet()) {
            if (!kv.getValue()) {
                throw new IllegalArgumentException(String.format("Response %s not found in file %s", kv.getKey(), csvPath));
            }
        }
    }

    private T buildOutput(String responseName, Map<String, String> row) {
        String label = row.get(responseName);
        T output = outputFactory.generateOutput(label);
        return output;
    }

    /**
     * Build a Output for a multi-output CSV file formatted like:
     *
     * Attr1,Attr2,...,Class1,Class2,Class3
     * 1.0,0.5,...,true,true,false
     * 1.0,0.5,...,true,false,false
     * 1.0,0.5,...,false,true,true
     *
     * Or for multivariate regression,
     *
     * Attr1,Attr2,...,Var1,Var2,Var3
     * 1.0,0.5,...,0.1,0.1,0.3
     * 1.0,0.5,...,0.2,0.0,0.8
     * @param responseNames The response dimension names
     * @param row The row to process.
     */
    private T buildMultiOutput(Set<String> responseNames, Map<String, String> row) {
        Set<String> pairs = new HashSet<>();
        for (String responseName : responseNames) {
            String rawValue = row.get(responseName);
            String pair = String.format("%s=%s", responseName, rawValue);
            pairs.add(pair);
        }
        T output = outputFactory.generateOutput(pairs);
        return output;
    }

    public final static class CSVLoaderProvenance implements DataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private static final String RESPONSE_NAME = "response-name";
        private static final String SEP_PROV = "separator";
        private static final String QUOTE_PROV = "quote";
        private static final String PATH = "path";

        private final StringProvenance className;
        private final OutputFactoryProvenance factoryProvenance;

        // In the multi-output case, the responseName will be a comma-separated list of response names
        private final StringProvenance responseName;
        private final CharProvenance separator;
        private final CharProvenance quote;
        private final URLProvenance path;
        private final DateTimeProvenance fileModifiedTime;
        private final HashProvenance sha256Hash;

        CSVLoaderProvenance(URL path, OutputFactoryProvenance factoryProvenance, String responseName, char separator, char quote) {
            this.className = new StringProvenance(CLASS_NAME,CSVLoader.class.getName());
            this.factoryProvenance = factoryProvenance;
            this.responseName = new StringProvenance(RESPONSE_NAME,responseName);
            this.separator = new CharProvenance(SEP_PROV,separator);
            this.quote = new CharProvenance(QUOTE_PROV,quote);
            this.path = new URLProvenance(PATH,path);
            Optional<OffsetDateTime> time = ProvenanceUtil.getModifiedTime(path);
            this.fileModifiedTime = time.map(offsetDateTime -> new DateTimeProvenance(FILE_MODIFIED_TIME, offsetDateTime)).orElseGet(() -> new DateTimeProvenance(FILE_MODIFIED_TIME, OffsetDateTime.MIN));
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE,RESOURCE_HASH, ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,path));
        }

        public CSVLoaderProvenance(Map<String, Provenance> map) {
            this.className = ObjectProvenance.checkAndExtractProvenance(map,CLASS_NAME,StringProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.factoryProvenance = ObjectProvenance.checkAndExtractProvenance(map,OUTPUT_FACTORY,OutputFactoryProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.responseName = ObjectProvenance.checkAndExtractProvenance(map,RESPONSE_NAME,StringProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.separator = ObjectProvenance.checkAndExtractProvenance(map,SEP_PROV,CharProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.quote = ObjectProvenance.checkAndExtractProvenance(map,QUOTE_PROV,CharProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.path = ObjectProvenance.checkAndExtractProvenance(map,PATH,URLProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.fileModifiedTime = ObjectProvenance.checkAndExtractProvenance(map,FILE_MODIFIED_TIME,DateTimeProvenance.class,CSVLoaderProvenance.class.getSimpleName());
            this.sha256Hash = ObjectProvenance.checkAndExtractProvenance(map,RESOURCE_HASH,HashProvenance.class,CSVLoaderProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return className.getValue();
        }

        @Override
        public Iterator<Pair<String, Provenance>> iterator() {
            ArrayList<Pair<String,Provenance>> list = new ArrayList<>();

            list.add(new Pair<>(CLASS_NAME,className));
            list.add(new Pair<>(OUTPUT_FACTORY,factoryProvenance));
            list.add(new Pair<>(RESPONSE_NAME,responseName));
            list.add(new Pair<>(SEP_PROV,separator));
            list.add(new Pair<>(QUOTE_PROV,quote));
            list.add(new Pair<>(PATH,path));
            list.add(new Pair<>(FILE_MODIFIED_TIME,fileModifiedTime));
            list.add(new Pair<>(RESOURCE_HASH,sha256Hash));

            return list.iterator();
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof CSVLoaderProvenance)) return false;
            CSVLoaderProvenance pairs = (CSVLoaderProvenance) o;
            return className.equals(pairs.className) &&
                    factoryProvenance.equals(pairs.factoryProvenance) &&
                    responseName.equals(pairs.responseName) &&
                    separator.equals(pairs.separator) &&
                    quote.equals(pairs.quote) &&
                    path.equals(pairs.path) &&
                    fileModifiedTime.equals(pairs.fileModifiedTime) &&
                    sha256Hash.equals(pairs.sha256Hash);
        }

        @Override
        public int hashCode() {
            return Objects.hash(className, factoryProvenance, responseName, separator, quote, path, fileModifiedTime, sha256Hash);
        }

        @Override
        public String toString() {
            return generateString("CSV");
        }
    }

}