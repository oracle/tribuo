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

package org.tribuo.data.columnar;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.config.PropertyException;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenancable;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.Model;
import org.tribuo.Output;
import org.tribuo.VariableInfo;
import org.tribuo.impl.ArrayExample;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * A processor which takes a Map of String to String and returns an {@link Example}.
 * <p>
 * It accepts a {@link ResponseProcessor} which converts the response field into an {@link Output}, a Map of
 * {@link FieldProcessor}s which converts fields into {@link ColumnarFeature}s, and a Set of {@link FeatureProcessor}s
 * which processes the list of {@link ColumnarFeature}s before {@link Example} construction. Optionally metadata and
 * weights can be extracted using {@link FieldExtractor}s and written into each example as they are constructed.
 * <p>
 * If the metadata extractors are invalid (i.e., two extractors write to the same metadata key), the RowProcessor throws
 * {@link PropertyException}.
 * @param <T> The output type.
 */
public class RowProcessor<T extends Output<T>> implements Configurable, Provenancable<ConfiguredObjectProvenance> {

    private static final Logger logger = Logger.getLogger(RowProcessor.class.getName());

    private static final String FEATURE_NAME_REGEX = "[" + ColumnarFeature.JOINER + FieldProcessor.NAMESPACE + "]";

    private static final Pattern FEATURE_NAME_PATTERN = Pattern.compile(FEATURE_NAME_REGEX);

    @Config(description = "Extractors for the example metadata.")
    private List<FieldExtractor<?>> metadataExtractors = Collections.emptyList();

    /**
     * The extractor for the example weight.
     */
    @Config(description = "Extractor for the example weight.")
    protected FieldExtractor<Float> weightExtractor = null;

    /**
     * The processor which extracts the response.
     */
    @Config(mandatory = true, description = "Processor which extracts the response.")
    protected ResponseProcessor<T> responseProcessor;

    @Config(mandatory = true, description = "The list of field processors to use.")
    private List<FieldProcessor> fieldProcessorList;

    /**
     * The map of field processors.
     */
    // fieldProcessorList is unpacked into this map to make the config files less complex.
    // fieldProcessorMap is the store of record for field processors.
    protected Map<String, FieldProcessor> fieldProcessorMap;

    @Config(description = "A set of feature processors to apply after extraction.")
    private Set<FeatureProcessor> featureProcessors = new HashSet<>();

    /**
     * The map of regexes to field processors.
     */
    @Config(description = "A map from a regex to field processors to apply to fields matching the regex.")
    protected Map<String, FieldProcessor> regexMappingProcessors = new HashMap<>();

    /**
     * Should newlines be replaced with spaces before processing.
     */
    @Config(description = "Replace newlines with spaces in values before passing them to field processors.")
    protected boolean replaceNewlinesWithSpaces = true;

    /**
     * Has this row processor been configured?
     */
    protected boolean configured;

    /**
     * Constructs a RowProcessor using the supplied responseProcessor to extract the response variable, and the supplied
     * fieldProcessorMap to control which fields are parsed and how they are parsed.
     * <p>
     * This processor does not generate any additional metadata for the examples, nor does it set the weight value on
     * generated examples.
     *
     * @param responseProcessor The response processor to use.
     * @param fieldProcessorMap The keys are the field names and the values are the field processors to apply to those
     *                          fields.
     */
    public RowProcessor(ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessorMap) {
        this(Collections.emptyList(), null, responseProcessor, fieldProcessorMap, Collections.emptySet());
    }

    /**
     * Constructs a RowProcessor using the supplied responseProcessor to extract the response variable, and the supplied
     * fieldProcessorMap to control which fields are parsed and how they are parsed.
     * <p>
     * After extraction the features are then processed using the supplied set of feature processors. These processors
     * can be used to insert conjunction features which are triggered when multiple features appear, or to filter out
     * unnecessary features.
     * <p>
     * This processor does not generate any additional metadata for the examples, nor does it set the weight value on
     * generated examples.
     *
     * @param responseProcessor The response processor to use.
     * @param fieldProcessorMap The keys are the field names and the values are the field processors to apply to those
     *                          fields.
     * @param featureProcessors The feature processors to run on each extracted feature list.
     */
    public RowProcessor(ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessorMap, Set<FeatureProcessor> featureProcessors) {
        this(Collections.emptyList(), null, responseProcessor, fieldProcessorMap, featureProcessors);
    }

    /**
     * Constructs a RowProcessor using the supplied responseProcessor to extract the response variable, and the supplied
     * fieldProcessorMap to control which fields are parsed and how they are parsed.
     * <p>
     * Additionally this processor can extract and populate metadata fields on the generated examples (e.g., the row
     * number, date stamps).
     *
     * @param metadataExtractors The metadata extractors to run per example. If two metadata extractors emit the same
     *                           metadata name then the constructor throws a PropertyException.
     * @param responseProcessor  The response processor to use.
     * @param fieldProcessorMap  The keys are the field names and the values are the field processors to apply to those
     *                           fields.
     * @deprecated Prefer {@link Builder} to many-argument constructors
     */
    @Deprecated
    public RowProcessor(List<FieldExtractor<?>> metadataExtractors, ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessorMap) {
        this(metadataExtractors, null, responseProcessor, fieldProcessorMap, Collections.emptySet());
    }

    /**
     * Constructs a RowProcessor using the supplied responseProcessor to extract the response variable, and the supplied
     * fieldProcessorMap to control which fields are parsed and how they are parsed.
     * <p>
     * After extraction the features are then processed using the supplied set of feature processors. These processors
     * can be used to insert conjunction features which are triggered when multiple features appear, or to filter out
     * unnecessary features.
     * <p>
     * Additionally this processor can extract a weight from each row and insert it into the example, along with more
     * general metadata fields (e.g., the row number, date stamps). The weightExtractor can be null, and if so the
     * weights are left unset.
     *
     * @param metadataExtractors The metadata extractors to run per example. If two metadata extractors emit the same
     *                           metadata name then the constructor throws a PropertyException.
     * @param weightExtractor    The weight extractor, if null the weights are left unset at their default.
     * @param responseProcessor  The response processor to use.
     * @param fieldProcessorMap  The keys are the field names and the values are the field processors to apply to those
     *                           fields.
     * @param featureProcessors  The feature processors to run on each extracted feature list.
     * @deprecated Prefer {@link Builder} to many-argument constructors
     */
    @Deprecated
    public RowProcessor(List<FieldExtractor<?>> metadataExtractors, FieldExtractor<Float> weightExtractor,
                        ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessorMap,
                        Set<FeatureProcessor> featureProcessors) {
        this(metadataExtractors, weightExtractor, responseProcessor, fieldProcessorMap, Collections.emptyMap(), featureProcessors, true);
    }

    /**
     * Constructs a RowProcessor using the supplied responseProcessor to extract the response variable, and the supplied
     * fieldProcessorMap to control which fields are parsed and how they are parsed.
     * <p>
     * In addition this processor can instantiate field processors which match the regexes supplied in the
     * regexMappingProcessors. If a regex matches a field which already has a fieldProcessor assigned to it, it throws
     * an IllegalArgumentException.
     * <p>
     * After extraction the features are then processed using the supplied set of feature processors. These processors
     * can be used to insert conjunction features which are triggered when multiple features appear, or to filter out
     * unnecessary features.
     * <p>
     * Additionally this processor can extract a weight from each row and insert it into the example, along with more
     * general metadata fields (e.g., the row number, date stamps). The weightExtractor can be null, and if so the
     * weights are left unset.
     *
     * @param metadataExtractors     The metadata extractors to run per example. If two metadata extractors emit the
     *                               same metadata name then the constructor throws a PropertyException.
     * @param weightExtractor        The weight extractor, if null the weights are left unset at their default.
     * @param responseProcessor      The response processor to use.
     * @param fieldProcessorMap      The keys are the field names and the values are the field processors to apply to
     *                               those fields.
     * @param regexMappingProcessors A set of field processors which can be instantiated if the regexes match the field
     *                               names.
     * @param featureProcessors      The feature processors to run on each extracted feature list.
     * @deprecated Prefer {@link Builder} to many-argument constructors
     */
    @Deprecated
    public RowProcessor(List<FieldExtractor<?>> metadataExtractors, FieldExtractor<Float> weightExtractor,
                        ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessorMap,
                        Map<String, FieldProcessor> regexMappingProcessors, Set<FeatureProcessor> featureProcessors) {
        this(metadataExtractors, weightExtractor, responseProcessor, fieldProcessorMap, regexMappingProcessors, featureProcessors, true);
    }

    /**
     * Constructs a RowProcessor using the supplied responseProcessor to extract the response variable, and the supplied
     * fieldProcessorMap to control which fields are parsed and how they are parsed.
     * <p>
     * In addition, this processor can instantiate field processors which match the regexes supplied in the
     * regexMappingProcessors. If a regex matches a field which already has a fieldProcessor assigned to it, it throws
     * an IllegalArgumentException.
     * <p>
     * After extraction the features are then processed using the supplied set of feature processors. These processors
     * can be used to insert conjunction features which are triggered when multiple features appear, or to filter out
     * unnecessary features.
     * <p>
     * Additionally, this processor can extract a weight from each row and insert it into the example, along with more
     * general metadata fields (e.g., the row number, date stamps). The weightExtractor can be null, and if so the
     * weights are left unset.
     *
     * @param metadataExtractors        The metadata extractors to run per example. If two metadata extractors emit the
     *                                  same metadata name then the constructor throws a PropertyException.
     * @param weightExtractor           The weight extractor, if null the weights are left unset at their default.
     * @param responseProcessor         The response processor to use.
     * @param fieldProcessorMap         The keys are the field names and the values are the field processors to apply to
     *                                  those fields.
     * @param regexMappingProcessors    A set of field processors which can be instantiated if the regexes match the
     *                                  field names.
     * @param featureProcessors         The feature processors to run on each extracted feature list.
     * @param replaceNewlinesWithSpaces Replace newlines with spaces in values before passing them to field processors.
     * @deprecated Prefer {@link Builder} to many-argument constructors
     */
    @Deprecated
    public RowProcessor(List<FieldExtractor<?>> metadataExtractors, FieldExtractor<Float> weightExtractor,
                        ResponseProcessor<T> responseProcessor, Map<String, FieldProcessor> fieldProcessorMap,
                        Map<String, FieldProcessor> regexMappingProcessors, Set<FeatureProcessor> featureProcessors,
                        boolean replaceNewlinesWithSpaces) {
        this.metadataExtractors = metadataExtractors.isEmpty() ? Collections.emptyList() : new ArrayList<>(metadataExtractors);
        this.weightExtractor = weightExtractor;
        this.responseProcessor = responseProcessor;
        this.fieldProcessorMap = new HashMap<>(fieldProcessorMap);
        this.regexMappingProcessors = regexMappingProcessors.isEmpty() ? Collections.emptyMap() : new HashMap<>(regexMappingProcessors);
        this.featureProcessors.addAll(featureProcessors);
        this.replaceNewlinesWithSpaces = replaceNewlinesWithSpaces;
        postConfig();
    }

    /**
     * For olcut.
     */
    protected RowProcessor() {
    }

    /**
     * Used by the OLCUT configuration system, and should not be called by external code.
     */
    @Override
    public void postConfig() {
        configured = regexMappingProcessors.isEmpty();
        if (fieldProcessorList != null) {
            fieldProcessorMap = fieldProcessorList.stream().collect(Collectors.toMap(FieldProcessor::getFieldName, Function.identity()));
        } else {
            fieldProcessorList = new ArrayList<>();
            fieldProcessorList.addAll(fieldProcessorMap.values());
        }
        Set<String> metadataNames = new HashSet<>();
        for (FieldExtractor<?> extractor : metadataExtractors) {
            String newMetadataName = extractor.getMetadataName();
            if (metadataNames.contains(newMetadataName)) {
                throw new PropertyException("", "metadataExtractors",
                        "Two metadata extractors found referencing the same metadata name '" + newMetadataName + "'");
            } else {
                metadataNames.add(newMetadataName);
            }
        }
    }

    /**
     * Returns the response processor this RowProcessor uses.
     *
     * @return The response processor.
     */
    public ResponseProcessor<T> getResponseProcessor() {
        return responseProcessor;
    }

    /**
     * Returns the map of {@link FieldProcessor}s this RowProcessor uses.
     *
     * @return The field processors.
     */
    public Map<String, FieldProcessor> getFieldProcessors() {
        return Collections.unmodifiableMap(fieldProcessorMap);
    }

    /**
     * Returns the set of {@link FeatureProcessor}s this RowProcessor uses.
     *
     * @return The feature processors.
     */
    public Set<FeatureProcessor> getFeatureProcessors() {
        return Collections.unmodifiableSet(featureProcessors);
    }

    /**
     * Generate an {@link Example} from the supplied row. Returns an empty Optional if there are no features, or the
     * response is required but it was not found. The latter case is used at training time.
     *
     * @param row            The row to process.
     * @param outputRequired If an Output must be found in the row to return an Example.
     * @return An Optional containing an Example if the row was valid, an empty Optional otherwise.
     */
    public Optional<Example<T>> generateExample(ColumnarIterator.Row row, boolean outputRequired) {

        List<String> responseValues = responseProcessor.getFieldNames().stream()
                .map(f -> row.getRowData().getOrDefault(f, ""))
                .collect(Collectors.toList());
        Optional<T> labelOpt = responseProcessor.process(responseValues);
        if (!labelOpt.isPresent() && outputRequired) {
            return Optional.empty();
        }

        List<ColumnarFeature> features = generateFeatures(row.getRowData());

        if (features.isEmpty()) {
            logger.warning(String.format("Row %d empty of features, omitting", row.getIndex()));
            return Optional.empty();
        } else {
            T label = labelOpt.orElse(responseProcessor.getOutputFactory().getUnknownOutput());

            Map<String, Object> metadata = generateMetadata(row);

            Example<T> example;
            if (weightExtractor == null) {
                example = new ArrayExample<>(label, metadata);
            } else {
                example = new ArrayExample<>(label,
                        weightExtractor.extract(row).orElse(Example.DEFAULT_WEIGHT),
                        metadata);
            }
            example.addAll(features);
            return Optional.of(example);
        }
    }

    /**
     * Generate an {@link Example} from the supplied row. Returns an empty Optional if there are no features, or the
     * response is required but it was not found.
     * <p>
     * Supplies -1 as the example index, used in cases where the index isn't meaningful.
     *
     * @param row            The row to process.
     * @param outputRequired If an Output must be found in the row to return an Example.
     * @return An Optional containing an Example if the row was valid, an empty Optional otherwise.
     */
    public Optional<Example<T>> generateExample(Map<String, String> row, boolean outputRequired) {
        return generateExample(-1, row, outputRequired);
    }

    /**
     * Generate an {@link Example} from the supplied row. Returns an empty Optional if there are no features, or the
     * response is required but it was not found. The latter case is used at training time.
     *
     * @param idx            The index for use in the example metadata if desired.
     * @param row            The row to process.
     * @param outputRequired If an Output must be found in the row to return an Example.
     * @return An Optional containing an Example if the row was valid, an empty Optional otherwise.
     */
    public Optional<Example<T>> generateExample(long idx, Map<String, String> row, boolean outputRequired) {
        return generateExample(new ColumnarIterator.Row(idx, new ArrayList<>(row.keySet()), row), outputRequired);
    }

    /**
     * Generates the example metadata from the supplied row and index.
     *
     * @param row The row to process.
     * @return A (possibly empty) map containing the metadata.
     */
    public Map<String, Object> generateMetadata(ColumnarIterator.Row row) {
        if (metadataExtractors.isEmpty()) {
            return Collections.emptyMap();
        } else {
            Map<String, Object> metadataMap = new HashMap<>();
            long idx = row.getIndex();

            for (FieldExtractor<?> field : metadataExtractors) {
                String metadataName = field.getMetadataName();
                Optional<?> extractedValue = field.extract(row);
                if (extractedValue.isPresent()) {
                    metadataMap.put(metadataName, extractedValue.get());
                } else {
                    logger.warning("Failed to extract field with name " + metadataName + " from index " + idx);
                }
            }

            return metadataMap;
        }
    }

    /**
     * Generates the features from the supplied row.
     *
     * @param row The row to process.
     * @return A (possibly empty) list of {@link ColumnarFeature}s.
     */
    public List<ColumnarFeature> generateFeatures(Map<String, String> row) {
        if (!configured) {
            throw new IllegalStateException("expandRegexMapping not called, yet there are entries in regexMappingProcessors which have not been bound to a field name.");
        }
        List<ColumnarFeature> features = new ArrayList<>();

        for (Map.Entry<String, FieldProcessor> e : fieldProcessorMap.entrySet()) {
            String value = row.get(e.getKey());
            if (value != null) {
                if (replaceNewlinesWithSpaces) {
                    value = value.replace('\n', ' ');
                }
                value = value.trim();
                features.addAll(e.getValue().process(value));
            }
        }

        for (FeatureProcessor f : featureProcessors) {
            features = f.process(features);
        }

        return features;
    }

    /**
     * The set of column names this will use for the feature processing.
     *
     * @return The set of column names it processes.
     */
    public Set<String> getColumnNames() {
        return Collections.unmodifiableSet(fieldProcessorMap.keySet());
    }

    /**
     * Returns a description of the row processor and it's fields.
     *
     * @return A String description of the RowProcessor.
     */
    public String getDescription() {
        String weightExtractorStr = weightExtractor == null ? "null" : weightExtractor.toString();
        if (configured || regexMappingProcessors.isEmpty()) {
            return "RowProcessor(responseProcessor=" + responseProcessor.toString() +
                    ",fieldProcessorMap=" + fieldProcessorMap.toString() +
                    ",featureProcessors=" + featureProcessors.toString() +
                    ",metadataExtractors=" + metadataExtractors.toString() +
                    ",weightExtractor=" + weightExtractorStr + ")";
        } else {
            return "RowProcessor(responseProcessor=" + responseProcessor.toString() +
                    ",fieldProcessorMap=" + fieldProcessorMap.toString() +
                    ",regexMappingProcessors=" + regexMappingProcessors.toString() +
                    ",featureProcessors=" + featureProcessors.toString() +
                    ",metadataExtractors=" + metadataExtractors.toString() +
                    ",weightExtractor=" + weightExtractorStr + ")";
        }
    }

    @Override
    public String toString() {
        return getDescription();
    }

    /**
     * Returns the metadata keys and value types that are extracted by this RowProcessor.
     *
     * @return The metadata keys and value types.
     */
    public Map<String, Class<?>> getMetadataTypes() {
        if (metadataExtractors.isEmpty()) {
            return Collections.emptyMap();
        } else {
            Map<String, Class<?>> types = new HashMap<>();

            for (FieldExtractor<?> extractor : metadataExtractors) {
                types.put(extractor.getMetadataName(), extractor.getValueType());
            }

            return types;
        }
    }

    /**
     * Returns true if the regexes have been expanded into field processors.
     *
     * @return True if the RowProcessor has seen the set of input fields.
     */
    public boolean isConfigured() {
        return configured;
    }

    /**
     * Uses similar logic to {@link org.tribuo.transform.TransformationMap#validateTransformations} to check the regexes
     * against the {@link ImmutableFeatureMap} contained in the supplied {@link Model}. Throws an
     * IllegalArgumentException if any regexes overlap with themselves, or with the currently defined set of
     * fieldProcessorMap.
     *
     * @param model The model to use to expand the regexes.
     */
    public void expandRegexMapping(Model<T> model) {
        expandRegexMapping(model.getFeatureIDMap());
    }

    /**
     * Uses similar logic to {@link org.tribuo.transform.TransformationMap#validateTransformations} to check the regexes
     * against the supplied feature map. Throws an IllegalArgumentException if any regexes overlap with themselves, or
     * with the currently defined set of fieldProcessorMap.
     *
     * @param featureMap The feature map to use to expand the regexes.
     */
    public void expandRegexMapping(ImmutableFeatureMap featureMap) {
        ArrayList<String> fieldNames = new ArrayList<>(featureMap.size());

        for (VariableInfo v : featureMap) {
            String[] split = FEATURE_NAME_PATTERN.split(v.getName(), 1);
            String fieldName = split[0];
            fieldNames.add(fieldName);
        }

        expandRegexMapping(fieldNames);
    }

    /**
     * Uses similar logic to {@link org.tribuo.transform.TransformationMap#validateTransformations} to check the regexes
     * against the supplied list of field names. Throws an IllegalArgumentException if any regexes overlap with
     * themselves, or with the currently defined set of fieldProcessorMap or if there are unmatched regexes after
     * processing.
     *
     * @param fieldNames The list of field names.
     */
    public void expandRegexMapping(Collection<String> fieldNames) {
        if (configured) {
            logger.warning("RowProcessor was already configured, yet expandRegexMapping was called with " + fieldNames.toString());
        } else {
            Set<String> regexesMatchingFieldNames = partialExpandRegexMapping(fieldNames);

            if (regexesMatchingFieldNames.size() != regexMappingProcessors.size()) {
                throw new IllegalArgumentException("Failed to match all the regexes, found " + regexesMatchingFieldNames.size() + ", required " + regexMappingProcessors.size());
            } else {
                regexMappingProcessors.clear();
                configured = true;
            }
        }
    }

    /**
     * Caveat Implementor! This method contains the logic of
     * {@link org.tribuo.data.columnar.RowProcessor#expandRegexMapping} without any of the checks that ensure the
     * RowProcessor is in a valid state. This can be overriden in a subclass to expand a regex mapping several times for
     * a single instance of RowProcessor. The caller is responsible for ensuring that fieldNames are not duplicated
     * within or between calls.
     *
     * @param fieldNames The list of field names - should contain only previously unseen field names.
     * @return the set of regexes that were matched by fieldNames.
     */
    protected Set<String> partialExpandRegexMapping(Collection<String> fieldNames) {
        HashSet<String> regexesMatchingFieldNames = new HashSet<>();
        // Loop over all regexes
        for (Map.Entry<String, FieldProcessor> e : regexMappingProcessors.entrySet()) {
            Pattern p = Pattern.compile(e.getKey());
            // Loop over all field names
            for (String s : fieldNames) {
                // Check if the pattern matches the field name
                if (p.matcher(s).matches()) {
                    // If it matches, add the field to the fieldProcessorMap map and the fieldProcessorList (for the provenance).
                    FieldProcessor newProcessor = e.getValue().copy(s);
                    fieldProcessorList.add(newProcessor);
                    FieldProcessor f = fieldProcessorMap.put(s, newProcessor);


                    if (f != null) {
                        throw new IllegalArgumentException("Regex " + p.toString() + " matched field " + s + " which already had a field processor " + f.toString());
                    }

                    regexesMatchingFieldNames.add(e.getKey());
                }
            }
        }
        return regexesMatchingFieldNames;
    }

    /**
     * @return a RowProcessor instance with clean state and the same configuration as this row processor.
     * @deprecated In a future release this API will change, in the meantime this is the correct way to get a row
     * processor with clean state.
     * <p>
     * When using regexMappingProcessors, RowProcessor is stateful in a way that can sometimes make it fail the second
     * time it is used. Concretely:
     * <pre>
     *     RowProcessor rp;
     *     Dataset ds1 = new MutableDataset(new CSVDataSource(csvfile1, rp));
     *     Dataset ds2 = new MutableDataset(new CSVDataSource(csvfile2, rp)); // this may fail due to state in rp
     * </pre>
     * This method returns a RowProcessor with clean state and the same configuration as this row processor.
     */
    @Deprecated
    public RowProcessor<T> copy() {
        return new RowProcessor<>(metadataExtractors, weightExtractor, responseProcessor, fieldProcessorMap, regexMappingProcessors, featureProcessors, replaceNewlinesWithSpaces);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "RowProcessor");
    }

    /**
     * Builder for {@link RowProcessor}.
     *
     * @param <T> The output type.
     */
    public static class Builder<T extends Output<T>> {

        private List<FieldExtractor<?>> metadataExtractors;
        private FieldExtractor<Float> weightExtractor;
        private Map<String, FieldProcessor> fieldProcessors;
        private Map<String, FieldProcessor> regexMappingProcessors;
        private Set<FeatureProcessor> featureProcessors;
        private boolean replaceNewLinesWithSpaces = true;

        /**
         * Builder for {@link RowProcessor}, see RowProcessor constructors for argument details.
         */
        public Builder() {
            metadataExtractors = new ArrayList<>();
            featureProcessors = new HashSet<>();
            regexMappingProcessors = new HashMap<>();
            fieldProcessors = new HashMap<>();
        }

        /**
         * If true, replaces newlines in fields with spaces before passing them to {@link FieldProcessor}s.
         * <p>
         * Defaults to true, some FieldProcessors may behave unexpectedly if this is false.
         *
         * @param replaceNewLinesWithSpaces Should newlines be replaced with spaces when parsing fields?
         * @return This builder with updated state
         */
        public Builder<T> setReplaceNewLinesWithSpaces(boolean replaceNewLinesWithSpaces) {
            this.replaceNewLinesWithSpaces = replaceNewLinesWithSpaces;
            return this;
        }

        /**
         * If set, the constructed {@link RowProcessor} will add the extracted floats into the
         * {@link Example#setWeight(float)}s. Otherwise, all examples will be equally weighted.
         *
         * @param weightExtractor FieldExtractor to generate example-wise weights
         * @return This builder with updated state
         */
        public Builder<T> setWeightExtractor(FieldExtractor<Float> weightExtractor) {
            this.weightExtractor = weightExtractor;
            return this;
        }

        /**
         * If set, the supplied {@link FieldExtractor}s will be run for each example, populating
         * {@link Example#getMetadata()}. If two FieldExtractors in this list share the same
         * {@link FieldExtractor#getMetadataName()}, {@link Builder#build(ResponseProcessor)} will throw a
         * {@link PropertyException}.
         * <p>
         * <strong>N.B.</strong> this method overrides all existing values for metadata extractors, to
         * non-destructively add use {@link #addMetadataExtractor(FieldExtractor)}.
         *
         * @param metadataExtractors List of metadata extractors to use
         * @return This builder with updated state
         */
        public Builder<T> setMetadataExtractors(List<FieldExtractor<?>> metadataExtractors) {
            this.metadataExtractors = metadataExtractors;
            return this;
        }

        /**
         * Add a single metadata extractor to the builder. See {@link #setMetadataExtractors(List)} for more detail.
         *
         * @param metadataExtractor The extractor to add.
         * @return This builder with updated state
         */
        public Builder<T> addMetadataExtractor(FieldExtractor<?> metadataExtractor) {
            metadataExtractors.add(metadataExtractor);
            return this;
        }

        /**
         * The {@link FeatureProcessor}s to apply to each extracted feature list.
         * <p>
         * <strong>N.B.</strong> this method overrides all existing values for feature processors, to non-destructively
         * add use {@link #addFeatureProcessor(FeatureProcessor)}.
         *
         * @param featureProcessors the processors to add.
         * @return This builder with updated state
         */
        public Builder<T> setFeatureProcessors(Set<FeatureProcessor> featureProcessors) {
            this.featureProcessors = featureProcessors;
            return this;
        }

        /**
         * Add a single feature processor to the builder. See {@link #setFeatureProcessors(Set)} for more detail.
         *
         * @param featureProcessor the processor to add.
         * @return This builder with updated state
         */
        public Builder<T> addFeatureProcessor(FeatureProcessor featureProcessor) {
            featureProcessors.add(featureProcessor);
            return this;
        }


        /**
         * Add a single field processor to the builder. If there is already a field processor with the same
         * {@link FieldProcessor#getFieldName()}, warns and overwrites.
         *
         * @param fieldProcessor The field processor to add
         * @return This builder with updated state
         */
        public Builder<T> addFieldProcessor(FieldProcessor fieldProcessor) {
            if (fieldProcessors.containsKey(fieldProcessor.getFieldName())) {
                logger.warning("Field name " + fieldProcessor.getFieldName() + " aleady present, overwriting");
            }
            fieldProcessors.put(fieldProcessor.getFieldName(), fieldProcessor);
            return this;
        }

        /**
         * the {@link FieldProcessor}s to apply to each row. Throws {@link IllegalArgumentException} if fieldProcessors
         * have duplicate {@link FieldProcessor#getFieldName()} values.
         * <p>
         * <strong>N.B.</strong> this method overrides all existing values for field processors, to non-descructively
         * add use {@link #addFieldProcessor(FieldProcessor)}.
         *
         * @param fieldProcessors an Iterable of field processors
         * @return This builder with updated state
         */
        public Builder<T> setFieldProcessors(Iterable<FieldProcessor> fieldProcessors) {
            Map<String, FieldProcessor> fpMap = new HashMap<>();
            for (FieldProcessor fieldProcessor : fieldProcessors) {
                if (fpMap.containsKey(fieldProcessor.getFieldName())) {
                    throw new IllegalArgumentException("Duplicate field name " + fieldProcessor.getFieldName());
                }
                fpMap.put(fieldProcessor.getFieldName(), fieldProcessor);
            }
            this.fieldProcessors = fpMap;
            return this;
        }

        /**
         * Retrieves, if present, the fieldProcessor with the given name
         * @param fieldName The field name.
         * @return The field processor or an empty optional.
         */
        public Optional<FieldProcessor> getFieldProcessor(String fieldName) {
            return Optional.ofNullable(this.fieldProcessors.get(fieldName));
        }

        /**
         * A map from strings (interpreted as regular expressions by {@link Pattern#compile(String)}) to
         * {@link FieldProcessor}s such that if a field name matches a regular expression, the corresponding
         * FieldProcessor is used to process it.
         * <p>
         * <strong>N.B.</strong> This method overrides all existing values for regex mapping, to non-destructively add
         * use {@link #addRegexMappingProcessor(String, FieldProcessor)}. If none of the keys in the regex mapping
         * processor match fields in your data, and there are only regex mapped fields, the {@link RowProcessor}'s
         * behavior is undefined.
         *
         * @param regexMappingProcessors The map from regex strings to FieldProcessors.
         * @return This builder with updated state
         */
        public Builder<T> setRegexMappingProcessors(Map<String, FieldProcessor> regexMappingProcessors) {
            this.regexMappingProcessors = regexMappingProcessors;
            return this;
        }

        /**
         * Retrieves, if present, the regexFieldProcessor with the given regex
         * @param regexName The regex.
         * @return The regexFieldProcessor or an empty optional.
         */
        public Optional<FieldProcessor> getRegexFieldProcessor(String regexName) {
            return Optional.ofNullable(this.regexMappingProcessors.get(regexName));
        }

        /**
         * Add a single regex FieldProcessor mapping to the builder. If there is already a field processor with the same
         * regex, warns and overwrites.
         * <p>
         * <strong>N.B.</strong> If none of the keys in the regex mapping processor match fields in your data, and
         * there are only regex mapped fields, the {@link RowProcessor}'s behavior is undefined.
         *
         * @param regex The regex to use.
         * @param fieldProcessor The field processor to apply when the regex matches.
         * @return This builder with updated state
         */
        public Builder<T> addRegexMappingProcessor(String regex, FieldProcessor fieldProcessor) {
            if (regexMappingProcessors.containsKey(regex)) {
                logger.warning("Regex pattern " + regex + " already present, overwriting");
            }
            regexMappingProcessors.put(regex, fieldProcessor);
            return this;
        }

        /**
         * Construct the {@link RowProcessor} represented by this builder's state. Throws {@link PropertyException} if
         * the state is invalid. Throws {@link IllegalArgumentException} if there is not at least one
         * {@link FieldProcessor} set via either {@link #addFieldProcessor(FieldProcessor)},
         * {@link #setFieldProcessors(Iterable)}, {@link #addRegexMappingProcessor(String, FieldProcessor)} or
         * {@link #setRegexMappingProcessors(Map)}.
         * <p>
         * <strong>N.B.</strong> If none of the keys in the regex mapping processor match fields in your data, and
         * there are only regex mapped fields, the {@link RowProcessor}'s behavior is undefined.
         *
         * @param responseProcessor The response processor to use.
         * @return The RowProcessor represented by the builder's state
         */
        public RowProcessor<T> build(ResponseProcessor<T> responseProcessor) {
            if (fieldProcessors.isEmpty() && regexMappingProcessors.isEmpty()) {
                throw new IllegalArgumentException("At least one FieldProcessor must be present");
            }
            return new RowProcessor<>(metadataExtractors, weightExtractor, responseProcessor, fieldProcessors, regexMappingProcessors, featureProcessors, replaceNewLinesWithSpaces);
        }
    }

}
