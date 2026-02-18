/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.protos;

import java.util.HashMap;
import java.util.Map;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;

/**
 * A cache used to dedupe objects during deserialization.
 * <p>
 * Supports deduping {@link ImmutableFeatureMap} and {@link ImmutableOutputInfo} instances as these are the largest
 * shared structures. Feature map instances which are equal according to {@link ImmutableFeatureMap#equals}
 * will be deduplicated by returning the canonical instance.
 */
public final class ProtoDeserializationCache {

    private final Map<ImmutableFeatureMap, ImmutableFeatureMap> featureMapCache;

    private final Map<ImmutableOutputInfo<?>, ImmutableOutputInfo<?>> outputInfoCache;

    /**
     * Create an empty deserialization cache.
     */
    public ProtoDeserializationCache() {
        this.featureMapCache = new HashMap<>(4);
        this.outputInfoCache = new HashMap<>(4);
    }

    /**
     * Canonicalises the feature map by checking if we've already seen this one, caching the value
     * if not, and returning an equal feature map.
     * @param featureMap The feature map to canonicalise.
     * @return The canonicalised value (i.e., a cached one which is equal to the method argument).
     */
    public ImmutableFeatureMap canonicalise(ImmutableFeatureMap featureMap) {
        ImmutableFeatureMap canonicalisedValue = featureMapCache.computeIfAbsent(featureMap, k -> featureMap);
        return canonicalisedValue;
    }

    /**
     * Canonicalises the output info by checking if we've already seen this one, caching the value
     * if not, and returning an equal output info.
     * @param outputInfo The output info to canonicalise.
     * @return The canonicalised value (i.e., a cached one which is equal to the method argument).
     */
    public ImmutableOutputInfo<?> canonicalise(ImmutableOutputInfo<?> outputInfo) {
        ImmutableOutputInfo<?> canonicalisedValue = outputInfoCache.computeIfAbsent(outputInfo, k -> outputInfo);
        return canonicalisedValue;
    }

    /**
     * Returns the current size of the output info cache.
     * @return The output info cache size.
     */
    public int outputInfoCacheSize() {
        return outputInfoCache.size();
    }

    /**
     * Returns the current size of the feature map cache.
     * @return The feature map cache size.
     */
    public int featureMapCacheSize() {
        return featureMapCache.size();
    }

}

