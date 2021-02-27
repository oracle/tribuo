package org.tribuo.data.text.impl;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.data.text.DocumentPreprocessor;

import java.util.logging.Logger;

/**
 * A document pre-processor for 20 newsgroup data. This processor will take a
 * news group message in a string and reduce it to the subject of the message
 * and the body of the message. It deals with a variety of weird conditions
 * (e.g., no headers, can't find subject, etc.)
 */
public class NewsPreprocessor implements DocumentPreprocessor {

    private static final Logger logger = Logger.getLogger(NewsPreprocessor.class.getName());

    private static final String subjHeader = "Subject: ";

    /**
     * Constructor.
     */
    public NewsPreprocessor() {}

    @Override
    public String processDoc(String doc) {
        //
        // Find the blank line separating the headers and the body.
        int sepInd = doc.indexOf("\n\n");
        if (sepInd < 0) {
            //
            // Didn't find one, so let's do nothing.
            return doc;
        }

        //
        // Find the subject header in the headers.
        int subjInd = doc.indexOf(subjHeader);
        if (subjInd < 0 || subjInd > sepInd) {
            //
            // No subject or it's past the header separator. Just send the body.
            return doc.substring(sepInd + 2);
        }
        //
        // Find the newline indicating the end of the subject.
        int subjEnd = doc.indexOf('\n', subjInd);
        if (subjEnd < 0 || subjEnd > sepInd) {
            return doc.substring(sepInd + 2);
        }

        return doc.substring(subjInd + subjHeader.length(), subjEnd) + '\n'
                + doc.substring(sepInd + 2);
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"DocumentPreprocessor");
    }
}
