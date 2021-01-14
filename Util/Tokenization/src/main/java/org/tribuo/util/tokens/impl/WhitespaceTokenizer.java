package org.tribuo.util.tokens.impl;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

public class WhitespaceTokenizer extends SplitFunctionTokenizer {

	public static SplitFunction whitespaceSplitCharacterFunction = (codepoint, index, cs) -> Character.isWhitespace(codepoint) ? SplitResult.SPLIT_AT : SplitResult.NO_SPLIT_WORD;
	
	public WhitespaceTokenizer() {
		super(whitespaceSplitCharacterFunction);
	}

	@Override
	public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
	}

    @Override
    public WhitespaceTokenizer clone() {
    	return new WhitespaceTokenizer(); 
    }

}
