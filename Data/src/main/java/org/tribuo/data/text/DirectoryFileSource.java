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

package org.tribuo.data.text;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.SkeletalConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import org.tribuo.ConfigurableDataSource;
import org.tribuo.Example;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.provenance.ConfiguredDataSourceProvenance;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.time.OffsetDateTime;
import java.time.ZoneId;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Queue;
import java.util.logging.Logger;

/**
 * A data source for a somewhat-common format for text classification datasets:
 * a top level directory that contains a number of subdirectories. Each of these
 * subdirectories contains the data for a output whose name is the name of the
 * subdirectory.
 * <p>
 * In these subdirectories are a number of files. Each file represents a single
 * document that should be labeled with the name of the subdirectory.
 * <p>
 * This data source will produce appropriately labeled {@code Examples<T>}
 * from each of these files.
 *
 * @param <T> The type of the features built by the underlying text processing
 * infrastructure.
 */
public class DirectoryFileSource<T extends Output<T>> implements ConfigurableDataSource<T> {

    private static final Logger logger = Logger.getLogger(DirectoryFileSource.class.getName());

    private static final Charset enc = StandardCharsets.UTF_8;

    /**
     * The top-level directory containing the data set.
     */
    @Config(description="The top-level directory containing the data set.")
    private Path dataDir = Paths.get(".");

    /**
     * Document preprocessors that should be run on the documents that make up
     * this data set.
     */
    @Config(description="The preprocessors to apply to the input documents.")
    protected List<DocumentPreprocessor> preprocessors = new ArrayList<>();

    /**
     * The factory that converts a String into an {@link Output}.
     */
    @Config(mandatory=true,description="The output factory to use.")
    protected OutputFactory<T> outputFactory;

    /**
     * The extractor that we'll use to turn text into examples.
     */
    @Config(mandatory=true,description="The feature extractor that converts text into examples.")
    protected TextFeatureExtractor<T> extractor;

    /**
     * for olcut
     */
    protected DirectoryFileSource() {}

    /**
     * Creates a data source that will use the given feature extractor and
     * document preprocessors on the data read from the files in the directories
     * representing classes.
     *
     * @param dataDir The directory to inspect.
     * @param outputFactory The output factory used to generate the outputs.
     * @param extractor The text feature extractor that will run on the
     * documents.
     * @param preprocessors Pre-processors that we will run on the documents
     * before extracting their features.
     */
    public DirectoryFileSource(Path dataDir, OutputFactory<T> outputFactory, TextFeatureExtractor<T> extractor, DocumentPreprocessor... preprocessors) {
        this.dataDir = dataDir;
        this.outputFactory = outputFactory;
        this.extractor = extractor;
        this.preprocessors.addAll(Arrays.asList(preprocessors));
    }

    @Override
    public String toString() {
        return "DirectoryDataSource(directory="+dataDir.toString()+",extractor="+extractor.toString()+",preprocessors="+preprocessors.toString()+")";
    }

    @Override
    public OutputFactory<T> getOutputFactory() {
        return outputFactory;
    }

    @Override
    public Iterator<Example<T>> iterator() {
        return new DirectoryIterator();
    }

    private class DirectoryIterator implements Iterator<Example<T>> {

        /**
         * The top-level paths in the provided directory, which is to say the
         * directories that give the labels their names.
         */
        private final Queue<Path> labelDirs = new ArrayDeque<>();

        /**
         * The path for the current output, resolved against the top-level
         * directory.
         */
        private Path labelPath;

        /**
         * The current output to apply to docs.
         */
        private String label;

        /**
         * The paths for the files in a particular output directory.
         */
        private final Queue<Path> labelPaths = new ArrayDeque<>();

        private final StringBuilder db = new StringBuilder();

        public DirectoryIterator() {
            //
            // Get the top-level paths AKA the tags.
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dataDir)) {
                for (Path entry : stream) {
                    labelDirs.offer(entry);
                }
            } catch (IOException ex) {
                throw new IllegalStateException("Can't open directory " + dataDir, ex);
            }
            logger.info(String.format("Got %d output directories in %s", labelDirs.size(), dataDir));
        }

        @Override
        public boolean hasNext() {
            if (labelPaths.isEmpty()) {
                return !labelDirs.isEmpty();
            } else {
                return true;
            }
        }

        @Override
        public Example<T> next() {
            if (labelPaths.isEmpty()) {
                if (labelDirs.isEmpty()) {
                    throw new NoSuchElementException("No more files");
                } else {
                    labelPath = labelDirs.poll();
                    label = labelPath.getFileName().toString();
                    try (DirectoryStream<Path> stream = Files.newDirectoryStream(labelPath)) {
                        for (Path entry : stream) {
                            labelPaths.offer(entry);
                        }
                        logger.info(String.format("Got %d paths in %s", labelPaths.size(), labelPath));
                    } catch (IOException ex) {
                        throw new IllegalStateException("Can't open directory " + labelPath, ex);
                    }
                }
            }
            Path p = labelPaths.poll();
            db.delete(0, db.length());
            try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(p.toFile()), enc))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) {
                        db.append('\n');
                    } else {
                        db.append(line);
                    }
                    db.append('\n');
                }
                String postproc = db.toString();
                for (DocumentPreprocessor preproc : preprocessors) {
                    postproc = preproc.processDoc(postproc);
                    if (postproc == null) {
                        break;
                    }
                }
                if (postproc != null) {
                    Example<T> ret = extractor.extract(outputFactory.generateOutput(label), postproc);
                    return ret;
                } else {
                    //
                    // Uh, it got post processed away. See if there's another one
                    // and return it.
                    if (!hasNext()) {
                        throw new NoSuchElementException("No more files");
                    }
                    return next();
                }
            } catch (IOException ex) {
                throw new IllegalStateException("Error reading path " + p, ex);
            }
        }

    }

    @Override
    public ConfiguredDataSourceProvenance getProvenance() {
        return new DirectoryFileSourceProvenance(this);
    }

    /**
     * Provenance for {@link DirectoryFileSource}.
     */
    public static class DirectoryFileSourceProvenance extends SkeletalConfiguredObjectProvenance implements ConfiguredDataSourceProvenance {
        private static final long serialVersionUID = 1L;

        private final DateTimeProvenance fileModifiedTime;
        private final DateTimeProvenance dataSourceCreationTime;

        <T extends Output<T>> DirectoryFileSourceProvenance(DirectoryFileSource<T> host) {
            super(host,"DataSource");
            this.fileModifiedTime = new DateTimeProvenance(FILE_MODIFIED_TIME,OffsetDateTime.ofInstant(Instant.ofEpochMilli(host.dataDir.toFile().lastModified()), ZoneId.systemDefault()));
            this.dataSourceCreationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,OffsetDateTime.now());
        }

        /**
         * Deserialization constructor.
         * @param map The provenances.
         */
        public DirectoryFileSourceProvenance(Map<String,Provenance> map) {
            this(extractProvenanceInfo(map));
        }

        private DirectoryFileSourceProvenance(ExtractedInfo info) {
            super(info);
            this.dataSourceCreationTime = (DateTimeProvenance) info.instanceValues.get(DATASOURCE_CREATION_TIME);
            this.fileModifiedTime = (DateTimeProvenance) info.instanceValues.get(FILE_MODIFIED_TIME);
        }

        /**
         * Splits the provenance into configured and non-configured values.
         * @param map the provenances.
         * @return The extracted information.
         */
        protected static ExtractedInfo extractProvenanceInfo(Map<String,Provenance> map) {
            Map<String,Provenance> configuredParameters = new HashMap<>(map);
            String className = ObjectProvenance.checkAndExtractProvenance(configuredParameters,CLASS_NAME, StringProvenance.class, DirectoryFileSourceProvenance.class.getSimpleName()).getValue();
            String hostTypeStringName = ObjectProvenance.checkAndExtractProvenance(configuredParameters,HOST_SHORT_NAME, StringProvenance.class, DirectoryFileSourceProvenance.class.getSimpleName()).getValue();

            Map<String,PrimitiveProvenance<?>> instanceParameters = new HashMap<>();
            instanceParameters.put(DATASOURCE_CREATION_TIME,ObjectProvenance.checkAndExtractProvenance(configuredParameters,DATASOURCE_CREATION_TIME,DateTimeProvenance.class, DirectoryFileSourceProvenance.class.getSimpleName()));

            return new ExtractedInfo(className,hostTypeStringName,configuredParameters,instanceParameters);
        }

        @Override
        public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
            Map<String,PrimitiveProvenance<?>> map = new HashMap<>();

            map.put(FILE_MODIFIED_TIME,fileModifiedTime);
            map.put(DATASOURCE_CREATION_TIME,dataSourceCreationTime);

            return map;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof DirectoryFileSourceProvenance)) return false;
            if (!super.equals(o)) return false;
            DirectoryFileSourceProvenance pairs = (DirectoryFileSourceProvenance) o;
            return fileModifiedTime.equals(pairs.fileModifiedTime) &&
                    dataSourceCreationTime.equals(pairs.dataSourceCreationTime);
        }

        @Override
        public int hashCode() {
            return Objects.hash(super.hashCode(), fileModifiedTime, dataSourceCreationTime);
        }
    }
}
