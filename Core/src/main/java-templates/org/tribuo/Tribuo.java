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
 * This class stores the current Tribuo version, along with other compile time information.
 */
public final class Tribuo {
    /**
     * The full Tribuo version string.
     */
    public static final String VERSION = "${project.version}";

    /**
     * The build timestamp.
     */
    public static final String BUILD_TIMESTAMP = "${maven.build.timestamp}";

    /**
     * The major version number.
     */
    public static final int MAJOR_VERSION;

    /**
     * The minor version number.
     */
    public static final int MINOR_VERSION;

    /**
     * The patch release number.
     */
    public static final int POINT_VERSION;

    /**
     * Any tag on the version number, e.g., SNAPSHOT, ALPHA, etc.
     */
    public static final String TAG_VERSION;

    /**
     * Is this a snapshot build.
     */
    public static final boolean IS_SNAPSHOT;

    static {
        String[] splitVersion = VERSION.split("\\.");
        MAJOR_VERSION = Integer.parseInt(splitVersion[0]);
        MINOR_VERSION = Integer.parseInt(splitVersion[1]);
        IS_SNAPSHOT = VERSION.contains("SNAPSHOT");
        String[] tags = splitVersion[2].split("-");
        POINT_VERSION = Integer.parseInt(tags[0]);
        if (tags.length > 1) {
            TAG_VERSION = tags[1];
        } else {
            TAG_VERSION = "";
        }
    }
}