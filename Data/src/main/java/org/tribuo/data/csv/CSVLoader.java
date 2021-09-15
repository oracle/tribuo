/*
 * Copyright (c) 2015-2021, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.DataSource;
import org.tribuo.MutableDataset;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.data.columnar.FieldProcessor;
import org.tribuo.data.columnar.ResponseProcessor;
import org.tribuo.data.columnar.RowProcessor;
import org.tribuo.data.columnar.processors.field.DoubleFieldProcessor;
import org.tribuo.data.columnar.processors.response.FieldResponseProcessor;
import org.tribuo.provenance.DataSourceProvenance;
import org.tribuo.provenance.OutputFactoryProvenance;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Load a DataSource/Dataset from a CSV file.
 * <p>
 * The delimiter and quote characters are user controlled, so this class can parse TSVs,
 * CSVs, semi-colon separated data and other types of single character delimiter separated data.
 * <p>
 * This class is a simple loader *only* for numerical CSV files with a String response field.
 * If you need more complex processing, the response field isn't present, or you don't wish to
 * use all of the columns as features then you should use {@link CSVDataSource} and build a
 * {@link org.tribuo.data.columnar.RowProcessor} to cope with your specific input format.
 * <p>
 * CSVLoader is thread safe and immutable.
 * @param <T> The type of the output generated.
 */
public class CSVLoader<T extends Output<T>> {

    private static final Logger logger = Logger.getLogger(CSVLoader.class.getName());

    private final char separator;
    private final char quote;
    private final OutputFactory<T> outputFactory;

    /**
     * Creates a CSVLoader using the supplied separator, quote and output factory.
     * @param separator The separator character.
     * @param quote The quote character.
     * @param outputFactory The output factory.
     */
    public CSVLoader(char separator, char quote, OutputFactory<T> outputFactory) {
        this.separator = separator;
        this.quote = quote;
        this.outputFactory = outputFactory;
    }

    /**
     * Creates a CSVLoader using the supplied separator and output factory.
     * Sets the quote to {@link CSVIterator#QUOTE}.
     * @param separator The separator character.
     * @param outputFactory The output factory.
     */
    public CSVLoader(char separator, OutputFactory<T> outputFactory) {
        this(separator, CSVIterator.QUOTE, outputFactory);
    }

    /**
     * Creates a CSVLoader using the supplied output factory.
     * Sets the separator to {@link CSVIterator#SEPARATOR} and the quote to {@link CSVIterator#QUOTE}.
     * @param outputFactory The output factory.
     */
    public CSVLoader(OutputFactory<T> outputFactory) {
        this(CSVIterator.SEPARATOR, CSVIterator.QUOTE, outputFactory);
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     *
     * @param csvPath      The path to load.
     * @param responseName The name of the response variable.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, String responseName) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseName));
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     *
     * @param csvPath      The path to load.
     * @param responseName The name of the response variable.
     * @param header       The header of the CSV if it's not present in the file.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, String responseName, String[] header) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseName, header));
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     * <p>
     * The {@code responseNames} set is traversed in iteration order to emit outputs,
     * and should be an ordered set to ensure reproducibility.
     *
     * @param csvPath       The path to load.
     * @param responseNames The names of the response variables.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, Set<String> responseNames) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseNames));
    }

    /**
     * Loads a DataSource from the specified csv file then wraps it in a dataset.
     * <p>
     * The {@code responseNames} set is traversed in iteration order to emit outputs,
     * and should be an ordered set to ensure reproducibility.
     *
     * @param csvPath       The path to load.
     * @param responseNames The names of the response variables.
     * @param header        The header of the CSV if it's not present in the file.
     * @return A dataset containing the csv data.
     * @throws IOException If the read failed.
     */
    public MutableDataset<T> load(Path csvPath, Set<String> responseNames, String[] header) throws IOException {
        return new MutableDataset<>(loadDataSource(csvPath, responseNames, header));
    }

    /**
     * Loads a DataSource from the specified csv path.
     *
     * @param csvPath      The csv to load from.
     * @param responseName The name of the response variable.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(Path csvPath, String responseName) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName));
    }

    /**
     * Loads a DataSource from the specified csv path.
     *
     * @param csvPath      The csv to load from.
     * @param responseName The name of the response variable.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(URL csvPath, String responseName) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName));
    }

    /**
     * Loads a DataSource from the specified csv path.
     *
     * @param csvPath      The csv to load from.
     * @param responseName The name of the response variable.
     * @param header       The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(Path csvPath, String responseName, String[] header) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName), header);
    }

    /**
     * Loads a DataSource from the specified csv path.
     *
     * @param csvPath      The csv to load from.
     * @param responseName The name of the response variable.
     * @param header       The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(URL csvPath, String responseName, String[] header) throws IOException {
        return loadDataSource(csvPath, Collections.singleton(responseName), header);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * <p>
     * The {@code responseNames} set is traversed in iteration order to emit outputs,
     * and should be an ordered set to ensure reproducibility.
     *
     * @param csvPath       The csv to load from.
     * @param responseNames The names of the response variables.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(Path csvPath, Set<String> responseNames) throws IOException {
        return loadDataSource(csvPath, responseNames, null);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * <p>
     * The {@code responseNames} set is traversed in iteration order to emit outputs,
     * and should be an ordered set to ensure reproducibility.
     *
     * @param csvPath       The csv to load from.
     * @param responseNames The names of the response variables.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(URL csvPath, Set<String> responseNames) throws IOException {
        return loadDataSource(csvPath, responseNames, null);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * <p>
     * The {@code responseNames} set is traversed in iteration order to emit outputs,
     * and should be an ordered set to ensure reproducibility.
     *
     * @param csvPath       The csv to load from.
     * @param responseNames The names of the response variables.
     * @param header        The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(Path csvPath, Set<String> responseNames, String[] header) throws IOException {
        return loadDataSource(csvPath.toUri().toURL(),responseNames,header);
    }

    /**
     * Loads a DataSource from the specified csv path.
     * <p>
     * The {@code responseNames} set is traversed in iteration order to emit outputs,
     * and should be an ordered set to ensure reproducibility.
     *
     * @param csvPath       The csv to load from.
     * @param responseNames The names of the response variables.
     * @param header        The header of the CSV if it's not present in the file.
     * @return A datasource containing the csv data.
     * @throws IOException If the disk read failed.
     */
    public DataSource<T> loadDataSource(URL csvPath, Set<String> responseNames, String[] header) throws IOException {
        List<String> headers = header == null || header.length == 0 ? Collections.emptyList() : Arrays.asList(header);
        URI csvURI;

        // Extract headers and convert to URI
        try (CSVIterator itr = new CSVIterator(csvPath.toURI(), separator, quote)) {
            List<String> extractedHeaders = itr.getFields();
            if (extractedHeaders.isEmpty() && headers.isEmpty()) {
                throw new IllegalArgumentException("Failed to read headers from CSV, and none were supplied.");
            }
            if (headers.size() != 0) {
                if (extractedHeaders.size() != headers.size()) {
                    throw new IllegalArgumentException("The csv contains " + extractedHeaders.size() + " fields, but only " + headers.size() + " headers were supplied.");
                }
            } else {
                headers = extractedHeaders;
            }
            csvURI = csvPath.toURI();
        } catch (URISyntaxException e) {
            throw new FileNotFoundException("Failed to read from URL '" + csvPath + "' as it could not be converted to a URI");
        }

        // Validate the responseNames
        if (responseNames.isEmpty()) {
            throw new IllegalArgumentException("At least one response name must be specified, but responseNames is empty.");
        }
        if (!headers.containsAll(responseNames)) {
            for (String s : responseNames) {
                if (!headers.contains(s)) {
                    throw new IllegalArgumentException(String.format("Response %s not found in file %s", s, csvPath));
                }
            }
        }

        // Construct the row processor
        Map<String, FieldProcessor> fieldProcessors = new HashMap<>();
        for (String field : headers) {
            if (!responseNames.contains(field)) {
                fieldProcessors.put(field,new DoubleFieldProcessor(field,true,true));
            }
        }
        boolean includeFieldName = responseNames.size() > 1;
        ResponseProcessor<T> responseProcessor = new FieldResponseProcessor<>(new ArrayList<>(responseNames),Collections.nCopies(responseNames.size(),""),outputFactory,includeFieldName,false);
        RowProcessor<T> rowProcessor = new RowProcessor<>(responseProcessor,fieldProcessors);

        if (header != null) {
            // if headers are supplied then we assume they aren't present in the csv file.
            return new CSVDataSource<>(csvURI, rowProcessor, true, separator, quote, headers);
        } else {
            return new CSVDataSource<>(csvURI, rowProcessor, true, separator, quote);
        }
    }

    /**
     * Provenance for CSVs loaded by {@link CSVLoader}.
     * @deprecated Deprecated in 4.2 as CSVLoader now returns a {@link CSVDataSource}.
     * This provenance is kept so older models can still load correctly.
     */
    @Deprecated
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
            this.className = new StringProvenance(CLASS_NAME, CSVLoader.class.getName());
            this.factoryProvenance = factoryProvenance;
            this.responseName = new StringProvenance(RESPONSE_NAME, responseName);
            this.separator = new CharProvenance(SEP_PROV, separator);
            this.quote = new CharProvenance(QUOTE_PROV, quote);
            this.path = new URLProvenance(PATH, path);
            Optional<OffsetDateTime> time = ProvenanceUtil.getModifiedTime(path);
            this.fileModifiedTime = time.map(offsetDateTime -> new DateTimeProvenance(FILE_MODIFIED_TIME, offsetDateTime)).orElseGet(() -> new DateTimeProvenance(FILE_MODIFIED_TIME, OffsetDateTime.MIN));
            this.sha256Hash = new HashProvenance(DEFAULT_HASH_TYPE, RESOURCE_HASH, ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE, path));
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public CSVLoaderProvenance(Map<String, Provenance> map) {
            this.className = ObjectProvenance.checkAndExtractProvenance(map, CLASS_NAME, StringProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.factoryProvenance = ObjectProvenance.checkAndExtractProvenance(map, OUTPUT_FACTORY, OutputFactoryProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.responseName = ObjectProvenance.checkAndExtractProvenance(map, RESPONSE_NAME, StringProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.separator = ObjectProvenance.checkAndExtractProvenance(map, SEP_PROV, CharProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.quote = ObjectProvenance.checkAndExtractProvenance(map, QUOTE_PROV, CharProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.path = ObjectProvenance.checkAndExtractProvenance(map, PATH, URLProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.fileModifiedTime = ObjectProvenance.checkAndExtractProvenance(map, FILE_MODIFIED_TIME, DateTimeProvenance.class, CSVLoaderProvenance.class.getSimpleName());
            this.sha256Hash = ObjectProvenance.checkAndExtractProvenance(map, RESOURCE_HASH, HashProvenance.class, CSVLoaderProvenance.class.getSimpleName());
        }

        @Override
        public String getClassName() {
            return className.getValue();
        }

        @Override
        public Iterator<Pair<String, Provenance>> iterator() {
            ArrayList<Pair<String, Provenance>> list = new ArrayList<>();

            list.add(new Pair<>(CLASS_NAME, className));
            list.add(new Pair<>(OUTPUT_FACTORY, factoryProvenance));
            list.add(new Pair<>(RESPONSE_NAME, responseName));
            list.add(new Pair<>(SEP_PROV, separator));
            list.add(new Pair<>(QUOTE_PROV, quote));
            list.add(new Pair<>(PATH, path));
            list.add(new Pair<>(FILE_MODIFIED_TIME, fileModifiedTime));
            list.add(new Pair<>(RESOURCE_HASH, sha256Hash));

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