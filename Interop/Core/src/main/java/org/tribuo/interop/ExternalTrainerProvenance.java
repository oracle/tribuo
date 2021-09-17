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

package org.tribuo.interop;

import com.oracle.labs.mlrg.olcut.provenance.ObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.PrimitiveProvenance;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.ProvenanceUtil;
import com.oracle.labs.mlrg.olcut.provenance.primitives.DateTimeProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.HashProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.URLProvenance;
import org.tribuo.Trainer;
import org.tribuo.provenance.TrainerProvenance;

import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * A dummy provenance for a model trained outside Tribuo.
 * <p>
 * It records the timestamp, hash and location of the loaded model.
 * <p>
 * If the model is already loaded into memory then the location is the current working directory and the timestamp
 * is when this class is instantiated.
 */
public final class ExternalTrainerProvenance implements TrainerProvenance {
    private static final long serialVersionUID = 1L;

    private final URLProvenance location;
    private final DateTimeProvenance fileModifiedTime;
    private final HashProvenance modelHash;

    /**
     * Creates an external trainer provenance, storing the location
     * and pulling in the timestamp and file hash.
     * @param location The location to use.
     */
    public ExternalTrainerProvenance(URL location) {
        this.location = new URLProvenance("location",location);
        Optional<OffsetDateTime> time = ProvenanceUtil.getModifiedTime(location);
        this.fileModifiedTime = time.map(offsetDateTime -> new DateTimeProvenance("fileModifiedTime", offsetDateTime)).orElseGet(() -> new DateTimeProvenance("fileModifiedTime", OffsetDateTime.MIN));
        this.modelHash = new HashProvenance(DEFAULT_HASH_TYPE,"modelHash", ProvenanceUtil.hashResource(DEFAULT_HASH_TYPE,location));
    }

    /**
     * Creates an external trainer provenance, computing the hash from the byte array and storing this
     * instant as the timestamp, and the current working directory as the location.
     * @param model The model as a byte array.
     */
    public ExternalTrainerProvenance(byte[] model) {
        try {
            this.location = new URLProvenance("location", Paths.get(".").toAbsolutePath().normalize().toUri().toURL());
        } catch (MalformedURLException e) {
            throw new IllegalStateException("Invalid URL for current working directory.",e);
        }
        this.fileModifiedTime = new DateTimeProvenance("fileModifiedTime", OffsetDateTime.now());
        this.modelHash = new HashProvenance(DEFAULT_HASH_TYPE,"modelHash", ProvenanceUtil.hashArray(DEFAULT_HASH_TYPE,model));
    }

    /**
     * Used by the provenance serialization system.
     * @param provenance The provenance to reconstruct.
     */
    public ExternalTrainerProvenance(Map<String,Provenance> provenance) {
        this.location = ObjectProvenance.checkAndExtractProvenance(provenance,"location",URLProvenance.class,ExternalTrainerProvenance.class.getSimpleName());
        this.fileModifiedTime = ObjectProvenance.checkAndExtractProvenance(provenance,"fileModifiedTime",DateTimeProvenance.class,ExternalTrainerProvenance.class.getSimpleName());
        this.modelHash = ObjectProvenance.checkAndExtractProvenance(provenance,"modelHash",HashProvenance.class,ExternalTrainerProvenance.class.getSimpleName());
    }

    @Override
    public Map<String, Provenance> getConfiguredParameters() {
        return Collections.emptyMap();
    }

    @Override
    public String getClassName() {
        return Trainer.class.getName();
    }

    @Override
    public String toString() {
        return generateString("ExternalTrainer");
    }

    @Override
    public Map<String, PrimitiveProvenance<?>> getInstanceValues() {
        Map<String,PrimitiveProvenance<?>> map = new HashMap<>();

        map.put(location.getKey(),location);
        map.put(fileModifiedTime.getKey(),fileModifiedTime);
        map.put(modelHash.getKey(),modelHash);

        return map;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExternalTrainerProvenance other = (ExternalTrainerProvenance) o;
        return location.equals(other.location) &&
                fileModifiedTime.equals(other.fileModifiedTime) &&
                modelHash.equals(other.modelHash);
    }

    @Override
    public int hashCode() {
        return Objects.hash(location, fileModifiedTime, modelHash);
    }
}
