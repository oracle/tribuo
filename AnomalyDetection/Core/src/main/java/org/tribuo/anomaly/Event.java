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

import org.tribuo.Output;

import java.util.Objects;

/**
 * An {@link Output} representing either an {@link EventType#ANOMALOUS} or an
 * {@link EventType#EXPECTED} event.
 * <p>
 * Event trainers are allowed to throw IllegalArgumentException if they are supplied
 * an {@link EventType#ANOMALOUS} at training time. It's noted in the documentation if they
 * do support training from anomalous and expected data.
 */
public final class Event implements Output<Event> {
    private static final long serialVersionUID = 1L;

    /**
     * The type of event.
     */
    public enum EventType {
        ANOMALOUS(1), EXPECTED(0), UNKNOWN(-1);
        private final int value;

        EventType(int value) {
            this.value = value;
        }

        protected int getID() {
            return value;
        }
    }

    private final EventType type;

    private final double score;

    public Event(EventType type, double score) {
        this.type = type;
        this.score = score;
    }

    public Event(EventType type) {
        this(type,Double.NaN);
    }

    /**
     * Get a real valued score for this label.
     *
     * If the score is not set then it returns Double.NaN.
     * @return The predicted score for this label.
     */
    public double getScore() {
        return score;
    }

    /**
     * Gets the event type.
     * @return An event type.
     */
    public EventType getType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Event)) return false;
        Event event = (Event) o;
        return Double.compare(event.score, score) == 0 &&
                type == event.type;
    }

    @Override
    public int hashCode() {
        return Objects.hash(type, score);
    }

    @Override
    public boolean fullEquals(Event o) {
        if (this == o) return true;
        if (o == null) return false;

        if ((!(Double.isNaN(o.score) && Double.isNaN(score))) && (Double.compare(o.score, score) != 0)) return false;
        return type != null ? type.equals(o.type) : o.type == null;
    }

    @Override
    public String toString() {
        if (Double.isNaN(score)) {
            return type.toString();
        } else {
            return "("+type.toString()+","+score+")";
        }
    }

    @Override
    public Event copy() {
        return new Event(type,score);
    }

    /**
     * Returns "EventType" or "EventType,score=eventScore".
     * @param includeConfidence Include whatever confidence score the label contains, if known.
     * @return A String form suitable for serialization.
     */
    @Override
    public String getSerializableForm(boolean includeConfidence) {
        if (includeConfidence && !Double.isNaN(score)) {
            return type.toString() + ",score=" + score;
        } else {
            return type.toString();
        }
    }

}
