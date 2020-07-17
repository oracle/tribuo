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

import org.tribuo.MutableOutputInfo;

/**
 * An {@link MutableOutputInfo} object for {@link Event}s.
 * <p>
 * Counts the number of {@link Event.EventType#ANOMALOUS}, {@link Event.EventType#EXPECTED}
 * and {@link Event.EventType#UNKNOWN} outputs observed. The unknown output is invalid
 * at training time, and used as a prediction time sentinel (similarly to other Tribuo
 * prediction tasks).
 * <p>
 * Anomaly detection has a fixed domain, so it will throw {@link IllegalArgumentException}
 * if you somehow modify the {@link Event.EventType} enum to add a new value.
 */
public final class MutableAnomalyInfo extends AnomalyInfo implements MutableOutputInfo<Event> {
    private static final long serialVersionUID = 1L;

    MutableAnomalyInfo() {
        super();
    }

    MutableAnomalyInfo(AnomalyInfo info) {
        super(info);
    }

    @Override
    public void observe(Event output) {
        if (output == AnomalyFactory.UNKNOWN_EVENT) {
            unknownCount++;
        } else {
            switch (output.getType()) {
                case ANOMALOUS:
                    anomalyCount++;
                    break;
                case EXPECTED:
                    expectedCount++;
                    break;
                default:
                    throw new IllegalArgumentException("Unexpected EventType, found " + output.getType());
            }
        }
    }

    @Override
    public void clear() {
        unknownCount = 0;
        anomalyCount = 0;
        expectedCount = 0;
    }

    @Override
    public MutableAnomalyInfo copy() {
        return new MutableAnomalyInfo(this);
    }
}
