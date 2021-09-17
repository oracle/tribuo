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

package org.tribuo.provenance;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.DataSource;
import org.tribuo.Output;
import org.tribuo.OutputFactory;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Objects;

/**
 * This class stores a String describing the data source, along with a
 * timestamp. It should not be used except for simple demos, or machine
 * created data. It is vastly preferable to create a {@link DataSource}
 * implementation with a specific provenance, rather than using this
 * to construct an empty {@link org.tribuo.MutableDataset}.
 */
public class SimpleDataSourceProvenance implements DataSourceProvenance {
    private static final long serialVersionUID = 1L;

    /**
     * The description field in the provenance.
     */
    public static final String DESCRIPTION = "description";

    private final String className = DataSource.class.getName();

    private final StringProvenance description;
    private final DateTimeProvenance creationTime;
    private final OutputFactoryProvenance outputFactoryProvenance;

    /**
     * This constructor initialises the provenance using the current time in the system timezone.
     * @param description The description of the data.
     * @param outputFactory The output factory used to process it.
     * @param <T> The type of the output.
     */
    public <T extends Output<T>> SimpleDataSourceProvenance(String description, OutputFactory<T> outputFactory) {
        this(description,OffsetDateTime.now(),outputFactory);
    }

    /**
     * This constructor initialises the provenance using the supplied description, time and output factory.
     * @param description The description of the data.
     * @param creationTime The time the data was created or processed.
     * @param outputFactory The output factory used to process it.
     * @param <T> The type of the output.
     */
    public <T extends Output<T>> SimpleDataSourceProvenance(String description, OffsetDateTime creationTime, OutputFactory<T> outputFactory) {
        this.description = new StringProvenance(DESCRIPTION,description);
        this.creationTime = new DateTimeProvenance(DATASOURCE_CREATION_TIME,creationTime);
        this.outputFactoryProvenance = outputFactory.getProvenance();
    }

    /**
     * Used for provenance deserialization.
     * @param map The provenance elements.
     */
    public SimpleDataSourceProvenance(Map<String,Provenance> map) {
        this.description = ObjectProvenance.checkAndExtractProvenance(map,DESCRIPTION,StringProvenance.class,SimpleDataSourceProvenance.class.getSimpleName());
        this.creationTime = ObjectProvenance.checkAndExtractProvenance(map,DATASOURCE_CREATION_TIME,DateTimeProvenance.class,SimpleDataSourceProvenance.class.getSimpleName());
        this.outputFactoryProvenance = ObjectProvenance.checkAndExtractProvenance(map,OUTPUT_FACTORY,OutputFactoryProvenance.class,SimpleDataSourceProvenance.class.getSimpleName());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SimpleDataSourceProvenance)) return false;
        SimpleDataSourceProvenance pairs = (SimpleDataSourceProvenance) o;
        return className.equals(pairs.className) &&
                description.equals(pairs.description) &&
                creationTime.equals(pairs.creationTime) &&
                outputFactoryProvenance.equals(pairs.outputFactoryProvenance);
    }

    @Override
    public int hashCode() {
        return Objects.hash(className, description, creationTime, outputFactoryProvenance);
    }

    @Override
    public String getClassName() {
        return className;
    }

    @Override
    public Iterator<Pair<String, Provenance>> iterator() {
        ArrayList<Pair<String,Provenance>> list = new ArrayList<>();

        list.add(new Pair<>(DESCRIPTION,description));
        list.add(new Pair<>(OUTPUT_FACTORY,outputFactoryProvenance));
        list.add(new Pair<>(DATASOURCE_CREATION_TIME,creationTime));

        return list.iterator();
    }

    @Override
    public String toString() {
        return generateString("DataSource");
    }
}
