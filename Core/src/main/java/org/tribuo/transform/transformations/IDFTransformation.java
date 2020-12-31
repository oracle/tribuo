package org.tribuo.transform.transformations;

import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import java.util.Collections;
import java.util.Map;
import org.tribuo.transform.TransformStatistics;
import org.tribuo.transform.Transformation;
import org.tribuo.transform.TransformationProvenance;
import org.tribuo.transform.Transformer;

/**
 * A feature transformation that computes the IDF for features and then transforms
 * them with a TF-IDF weighting.
 */
public class IDFTransformation implements Transformation {
    
    private TransformationProvenance provenance;

    @Override
    public TransformStatistics createStats() {
        return new IDFStatistics();
    }

    @Override
    public TransformationProvenance getProvenance() {
        if (provenance == null) {
            provenance = new IDFTransformationProvenance();
        }
        return provenance;
    }
    
    private static class IDFStatistics implements TransformStatistics {
        
        /**
         * The document frequency for the feature that this statistic is
         * associated with. This is a count of the number of examples that the
         * feature occurs in.
         */
        private int df;
        
        /**
         * The number of examples that the feature did not occur in.
         */
        private int sparseObservances;
        

        @Override
        public void observeValue(double value) {
            //
            // One more document (i.e., an example) has this feature.
            df++;
        }

        @Override
        public void observeSparse() {
            sparseObservances++;
        }

        @Override
        public void observeSparse(int count) {
            sparseObservances = count;
        }

        @Override
        public Transformer generateTransformer() {
            return new IDFTransformer(df, df+sparseObservances);
        }
        
    }
    
    private static class IDFTransformer implements Transformer {
        private static final long serialVersionUID = 1L;

        private double df;
        
        private double N;
        
        public IDFTransformer(int df, int N) {
            this.df = df;
            this.N = N;
        }

        @Override
        public double transform(double tf) {
            return Math.log(N / df) * (1 + Math.log(tf));
        }
        
    }
    
    public final static class IDFTransformationProvenance implements TransformationProvenance {
        private static final long serialVersionUID = 1L;

        @Override
        public Map<String, Provenance> getConfiguredParameters() {
            return Collections.emptyMap();
        }

        @Override
        public String getClassName() {
            return IDFTransformation.class.getName();
        }
        
    }

}
