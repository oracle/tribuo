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

package org.tribuo;

/**
 * Same as a {@link RealInfo}, but with an additional int id field.
 */
public class RealIDInfo extends RealInfo implements VariableIDInfo {
    private static final long serialVersionUID = 1L;

    private final int id;

    public RealIDInfo(String name, int count, double max, double min, double mean, double sumSquares, int id) {
        super(name,count,max,min,mean,sumSquares);
        this.id = id;
    }

    public RealIDInfo(RealInfo info, int id) {
        super(info);
        this.id = id;
    }

    private RealIDInfo(RealIDInfo info, String newName) {
        super(info,newName);
        this.id = info.id;
    }

    @Override
    public int getID() {
        return id;
    }

    @Override
    public RealIDInfo makeIDInfo(int id) {
        return new RealIDInfo(this,id);
    }

    @Override
    public RealIDInfo rename(String newName) {
        return new RealIDInfo(this,newName);
    }

    @Override
    public RealIDInfo copy() {
        return new RealIDInfo(this,name);
    }

    @Override
    public String toString() {
        return String.format("RealFeature(name=%s,id=%d,count=%d,max=%f,min=%f,mean=%f,variance=%f)",name,id,count,max,min,mean,(sumSquares /(count-1)));
    }
}
