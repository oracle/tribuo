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

package org.tribuo.multilabel;

import com.oracle.labs.mlrg.olcut.util.MutableLong;
import org.tribuo.MutableOutputInfo;

import java.util.Map;

/**
 * A MutableOutputInfo for working with multi-label tasks.
 */
public class MutableMultiLabelInfo extends MultiLabelInfo implements MutableOutputInfo<MultiLabel> {
    private static final long serialVersionUID = 1L;

    MutableMultiLabelInfo() {
        super();
    }

    public MutableMultiLabelInfo(MultiLabelInfo info) {
        super(info);
    }

    /**
     * Throws IllegalStateException if the MultiLabel contains a Label which has a "," in it.
     * <p>
     * Such labels are disallowed. There should be an exception thrown when one is constructed
     * too.
     * @param output The observed output.
     */
    @Override
    public void observe(MultiLabel output) {
        if (output == MultiLabelFactory.UNKNOWN_MULTILABEL) {
            unknownCount++;
        } else {
            for (String label : output.getNameSet()) {
                if (label.contains(",")) {
                    throw new IllegalStateException("MultiLabel cannot use a Label which contains ','. The supplied label was " + label + ".");
                }
                MutableLong value = labelCounts.computeIfAbsent(label, k -> new MutableLong());
                labels.computeIfAbsent(label, MultiLabel::new);
                value.increment();
            }
            totalCount++;
        }
    }

    @Override
    public void clear() {
        labelCounts.clear();
        totalCount = 0;
    }

    @Override
    public MutableMultiLabelInfo copy() {
        return new MutableMultiLabelInfo(this);
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
}
