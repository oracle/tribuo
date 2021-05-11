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

package org.tribuo.classification;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;

import java.util.Map;

/**
 * A mutable {@link LabelInfo}. Can record new observations of Labels, incrementing the
 * appropriate counts.
 */
public class MutableLabelInfo extends LabelInfo implements MutableOutputInfo<Label> {
    private static final long serialVersionUID = 1L;

    MutableLabelInfo() {
        super();
    }

    /**
     * Constructs a mutable deep copy of the supplied label info.
     * @param info The info to copy.
     */
    public MutableLabelInfo(LabelInfo info) {
        super(info);
    }

    @Override
    public void observe(Label output) {
        if (output == LabelFactory.UNKNOWN_LABEL) {
            unknownCount++;
        } else {
            String label = output.getLabel();
            MutableLong value = labelCounts.computeIfAbsent(label, k -> new MutableLong());
            labels.computeIfAbsent(label, Label::new);
            value.increment();
        }
    }

    @Override
    public void clear() {
        labelCounts.clear();
    }

    @Override
    public MutableLabelInfo copy() {
        return new MutableLabelInfo(this);
    }

    @Override
    public String toReadableString() {
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<String,MutableLong> e : labelCounts.entrySet()) {
            if (builder.length() > 0) {
                builder.append(", ");
            }
            builder.append('(');
            builder.append(e.getKey());
            builder.append(',');
            builder.append(e.getValue().longValue());
            builder.append(')');
        }
        return builder.toString();
    }

    @Override
    public String toString() {
        return toReadableString();
    }
}
