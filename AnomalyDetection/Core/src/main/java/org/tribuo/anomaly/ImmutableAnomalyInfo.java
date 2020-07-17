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
import org.tribuo.anomaly.Event.EventType;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An {@link ImmutableOutputInfo} object for {@link Event}s.
 * <p>
 * The ids are predefined for {@link Event} in the Event class itself.
 */
public final class ImmutableAnomalyInfo extends AnomalyInfo implements ImmutableOutputInfo<Event> {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ImmutableAnomalyInfo.class.getName());

    ImmutableAnomalyInfo(AnomalyInfo info) {
        super(info);
    }

    @Override
    public int getID(Event output) {
        return output.getType().getID();
    }

    @Override
    public Event getOutput(int id) {
        if (id == EventType.ANOMALOUS.getID()) {
            return AnomalyFactory.ANOMALOUS_EVENT;
        } else if (id == EventType.EXPECTED.getID()) {
            return AnomalyFactory.EXPECTED_EVENT;
        } else {
            logger.log(Level.INFO,"No entry found for id " + id);
            return null;
        }
    }

    @Override
    public long getTotalObservations() {
        return anomalyCount + expectedCount;
    }

    @Override
    public ImmutableAnomalyInfo copy() {
        return new ImmutableAnomalyInfo(this);
    }

    @Override
    public Iterator<Pair<Integer, Event>> iterator() {
        List<Pair<Integer,Event>> list = new ArrayList<>();

        list.add(new Pair<>(AnomalyFactory.ANOMALOUS_EVENT.getType().getID(),AnomalyFactory.ANOMALOUS_EVENT));
        list.add(new Pair<>(AnomalyFactory.EXPECTED_EVENT.getType().getID(),AnomalyFactory.EXPECTED_EVENT));

        return list.iterator();
    }
}
