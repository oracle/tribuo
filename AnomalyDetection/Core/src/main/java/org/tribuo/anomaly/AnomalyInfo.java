/*
 * Copyright (c) 2015, 2022, Oracle and/or its affiliates. All rights reserved.
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
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.OutputDomainProto;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * The base class for tracking anomalous events.
 */
public abstract class AnomalyInfo implements OutputInfo<Event>  {
    private static final long serialVersionUID = 1L;

    private static final Set<Event> DOMAIN = makeDomain();

    /**
     * The number of expected events observed.
     */
    @ProtoSerializableField
    protected long expectedCount = 0;

    /**
     * The number of anomalous events observed.
     */
    @ProtoSerializableField
    protected long anomalyCount = 0;

    /**
     * The number of unknown events observed (i.e., those without labels).
     */
    @ProtoSerializableField
    protected int unknownCount = 0;

    /**
     * Constructs a new empty anomaly info.
     */
    protected AnomalyInfo() { }

    /**
     * Copies the supplied anomaly info.
     * @param other The info to copy.
     */
    protected AnomalyInfo(AnomalyInfo other) {
        this.expectedCount = other.expectedCount;
        this.anomalyCount = other.anomalyCount;
        this.unknownCount = other.unknownCount;
    }

    /**
     * Deserialization constructor.
     * <p>
     * Validates that the inputs are non-negative.
     * @param expectedCount The observed number of expected events.
     * @param anomalyCount The observed number of anomalous events.
     * @param unknownCount The observed number of unknown events.
     */
    protected AnomalyInfo(long expectedCount, long anomalyCount, int unknownCount) {
        if (expectedCount < 0) {
            throw new IllegalStateException("Invalid expectedCount, found " + expectedCount);
        }
        if (anomalyCount < 0) {
            throw new IllegalStateException("Invalid anomalyCount, found " + anomalyCount);
        }
        if (unknownCount < 0) {
            throw new IllegalStateException("Invalid unknownCount, found " + unknownCount);
        }
        this.expectedCount = expectedCount;
        this.anomalyCount = anomalyCount;
        this.unknownCount = unknownCount;
    }

    @Override
    public OutputDomainProto serialize() {
        return ProtoUtil.serialize(this);
    }

    @Override
    public int getUnknownCount() {
        return unknownCount;
    }

    /**
     * The number of anomalous events observed.
     * @return The number of anomalies.
     */
    public long getAnomalyCount() {
        return anomalyCount;
    }

    /**
     * The number of expected events observed.
     * @return The number of normal events.
     */
    public long getExpectedCount() {
        return expectedCount;
    }

    /**
     * Returns the set of possible {@link Event}s.
     * <p>
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
     * The number of possible event types (i.e., 2).
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
        return "AnomalyInfo(anomalies="+anomalyCount+",expected="+expectedCount+",unknown="+unknownCount+")";
    }

    @Override
    public String toString() {
        return toReadableString();
    }

    @Override
    public abstract AnomalyInfo copy();

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        AnomalyInfo that = (AnomalyInfo) o;
        return expectedCount == that.expectedCount && anomalyCount == that.anomalyCount && unknownCount == that.unknownCount;
    }

    @Override
    public int hashCode() {
        return Objects.hash(expectedCount, anomalyCount, unknownCount);
    }

    private static Set<Event> makeDomain() {
        HashSet<Event> set = new HashSet<>();

        set.add(AnomalyFactory.EXPECTED_EVENT);
        set.add(AnomalyFactory.ANOMALOUS_EVENT);

        return Collections.unmodifiableSet(set);
    }
}
