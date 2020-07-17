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

package org.tribuo.anomaly;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.MutableOutputInfo;
import org.tribuo.OutputInfo;
import org.tribuo.anomaly.Event.EventType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The base class for tracking anomalous events.
 */
public abstract class AnomalyInfo implements OutputInfo<Event>  {
    private static final long serialVersionUID = 1L;

    private static final Set<Event> DOMAIN = makeDomain();

    protected long expectedCount = 0;
    protected long anomalyCount = 0;
    protected int unknownCount = 0;

    protected AnomalyInfo() { }

    protected AnomalyInfo(AnomalyInfo other) {
        this.expectedCount = other.expectedCount;
        this.anomalyCount = other.anomalyCount;
        this.unknownCount = other.unknownCount;
    }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    public long getAnomalyCount() {
        return anomalyCount;
    }

    public long getExpectedCount() {
        return expectedCount;
    }

    /**
     * Returns the set of possible {@link Event}s.
     *
     * Each event has the default score of Double.NaN.
     * @return The set of possible events.
     */
    @Override
    public Set<Event> getDomain() {
        return DOMAIN;
    }

    /**
     * Gets the count of the supplied EventType.
     * @param type An EventType.
     * @return A non-negative long.
     */
    public long getEventCount(EventType type) {
        switch (type) {
            case ANOMALOUS:
                return anomalyCount;
            case EXPECTED:
                return expectedCount;
            case UNKNOWN:
                return unknownCount;
            default:
                return 0;
        }
    }

    @Override
    public Iterable<Pair<String,Long>> outputCountsIterable() {
        List<Pair<String,Long>> list = new ArrayList<>();

        list.add(new Pair<>(EventType.ANOMALOUS.toString(),anomalyCount));
        list.add(new Pair<>(EventType.EXPECTED.toString(),expectedCount));

        return list;
    }

    /**
     * The number of possible event types (i.e. 2).
     * @return The number of possible event types.
     */
    @Override
    public int size() {
        return DOMAIN.size();
    }

    @Override
    public ImmutableOutputInfo<Event> generateImmutableOutputInfo() {
        return new ImmutableAnomalyInfo(this);
    }

    @Override
    public MutableOutputInfo<Event> generateMutableOutputInfo() {
        return new MutableAnomalyInfo(this);
    }

    @Override
    public String toReadableString() {
        return "{Anomalies:"+anomalyCount+",expected:"+expectedCount+"}";
    }

    @Override
    public abstract AnomalyInfo copy();

    private static Set<Event> makeDomain() {
        HashSet<Event> set = new HashSet<>();

        set.add(AnomalyFactory.EXPECTED_EVENT);
        set.add(AnomalyFactory.ANOMALOUS_EVENT);

        return Collections.unmodifiableSet(set);
    }
}
