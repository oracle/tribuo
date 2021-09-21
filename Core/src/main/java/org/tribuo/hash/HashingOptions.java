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

package org.tribuo.hash;

import com.oracle.labs.mlrg.olcut.config.Option;
import com.oracle.labs.mlrg.olcut.config.Options;
import org.tribuo.Output;
import org.tribuo.Trainer;

import java.util.Optional;
import java.util.logging.Logger;

/**
 * An Options implementation which provides CLI arguments for the model hashing functionality.
 */
public class HashingOptions implements Options {
    private static final Logger logger = Logger.getLogger(HashingOptions.class.getName());

    /**
     * Supported types of hashes in CLI programs.
     */
    public enum ModelHashingType {
        /**
         * No hashing applied.
         */
        NONE,
        /**
         * Takes the String hash code mod some value.
         */
        MOD,
        /**
         * Uses the String hash code.
         */
        HC,
        /**
         * Uses SHA-1.
         */
        SHA1,
        /**
         * Uses SHA-256.
         */
        SHA256
    }

    /**
     * Hash the model during training, options are {NONE,MOD,HC,SHA1,SHA256}
     */
    @Option(longName = "model-hashing-algorithm", usage = "Hash the model during training, options are {NONE,MOD,HC,SHA1,SHA256}")
    public ModelHashingType modelHashingAlgorithm = ModelHashingType.NONE;
    /**
     * Salt for hashing the model
     */
    @Option(longName = "model-hashing-salt", usage = "Salt for hashing the model")
    public String modelHashingSalt = "";

    /**
     * Get the specified hasher.
     *
     * @return The configured hasher.
     */
    public Optional<Hasher> getHasher() {
        if (modelHashingAlgorithm == ModelHashingType.NONE) {
            return Optional.empty();
        } else if (Hasher.validateSalt(modelHashingSalt)) {
            switch (modelHashingAlgorithm) {
                case MOD:
                    return Optional.of(new ModHashCodeHasher(modelHashingSalt));
                case HC:
                    return Optional.of(new HashCodeHasher(modelHashingSalt));
                case SHA1:
                    return Optional.of(new MessageDigestHasher("SHA1", modelHashingSalt));
                case SHA256:
                    return Optional.of(new MessageDigestHasher("SHA-256", modelHashingSalt));
                default:
                    logger.info("Unknown hasher " + modelHashingAlgorithm);
                    return Optional.empty();
            }
        } else {
            logger.info("Invalid salt");
            return Optional.empty();
        }
    }

    /**
     * Gets the trainer wrapped in a hashing trainer.
     *
     * @param innerTrainer The inner trainer.
     * @param <T>          The output type.
     * @return The hashing trainer.
     */
    public <T extends Output<T>> Trainer<T> getHashedTrainer(Trainer<T> innerTrainer) {
        Optional<Hasher> hasherOpt = getHasher();
        if (hasherOpt.isPresent()) {
            return new HashingTrainer<>(innerTrainer, hasherOpt.get());
        } else {
            throw new IllegalArgumentException("Invalid Hasher");
        }
    }
}
