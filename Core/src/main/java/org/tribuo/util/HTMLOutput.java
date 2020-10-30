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

package org.tribuo.util;

import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Model;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utilities for nice HTML output that can be put in wikis and such.
 */
public final class HTMLOutput {

    private HTMLOutput() { }

    /**
     * Formats a pair as a HTML table entry.
     * @param p The pair to format.
     * @return A string containing the HTML representation of the input.
     */
    public static String toHTML(Pair<String, Double> p) {
        String cleanName = p.getA().replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;");

        return String.format("<td style=\"text-align:left\">%s</td><td style=\"text-align:right\">%.3f</td>",
                cleanName, p.getB());
    }

    /**
     * Formats a feature ranking as a HTML table.
     * @param m The ranking to format.
     * @param keys The classes to format.
     * @param s The stream to write to.
     */
    public static void printFeatureMap(Map<String, List<Pair<String, Double>>> m, List<String> keys, PrintStream s) {
        List<String> realKeys = new ArrayList<>(keys);
        realKeys.add(Model.ALL_OUTPUTS);
        if (m == null) {
            return;
        }
        s.println("<table>\n<tr>");
        for (String k : realKeys) {
            if (m.containsKey(k)) {
                s.printf("<th colspan=\"2\">%s</th>", k);
            }
        }
        s.print("</tr>\n<tr>");
        for (String k : realKeys) {
            if (m.containsKey(k)) {
                s.print("<th>Feature</th><th>Weight</th>");
            }
        }
        s.println("</tr>");
        //
        // We'll go until all classes are out of features.
        Map<String, Integer> pos = new HashMap<>();
        int done = 0;
        while (done < m.size()) {
            s.print("<tr>");
            for (String k : realKeys) {
                List<Pair<String, Double>> l = m.get(k);
                if (l == null) {
                    continue;
                }
                Integer p = pos.getOrDefault(k, 0);
                if (p >= l.size()) {
                    //
                    // We might be out of features for this class, so print an 
                    // empty one.
                    s.print("<td></td><td></td>");
                    continue;
                }
                s.print(toHTML(l.get(p)));
                p++;
                if (p >= l.size()) {
                    done++;
                }
                pos.put(k, p);
            }
            s.println("</tr>");
        }
        s.println("</table>");
    }

}
