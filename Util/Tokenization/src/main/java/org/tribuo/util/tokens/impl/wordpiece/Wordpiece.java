package org.tribuo.util.tokens.impl.wordpiece;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.config.Configurable;
import com.oracle.labs.mlrg.olcut.util.IOUtil;

/**
 * This is vanilla implementation of the Wordpiece algorithm as found here:
 * 
 * <a href=
 * "https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py">
 * https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py</a>
 * 
 * <p>
 * Please refer to the class definition for <code>WordpieceTokenizer</code>.
 * It does not include any of the tokenization work that is typically performed
 * before wordpiece is called as is done in the above-referenced implementation.
 * That functionality is provided by {@link WordpieceTokenizer} and
 * {@link WordpiecePreprocessTokenizer}.
 * 
 */
public class Wordpiece {

	private Set<String> vocab;
	private String unknownToken;
	private int maxInputCharactersPerWord;

	public Wordpiece(Set<String> vocab) {
		this(vocab, "[UNK]");
	}

	public Wordpiece(Set<String> vocab, String unknownToken) {
		this(vocab, unknownToken, 100);
	}

	/**
	 * Initializes an instance of Wordpiece with the given vocabulary, unknown token, and max word length.
	 * @param vocab the pre-trained wordpiece vocabulary.  See the contents of e.g. https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt
	 * @param unknownToken a string used to indicate a token was not found in the vocabulary - typically "[UNK]"
	 * @param maxInputCharactersPerWord a maximum to shield against looping over character-by-character pathologically long "tokens"
	 */
	public Wordpiece(Set<String> vocab, String unknownToken, int maxInputCharactersPerWord) {
		this.vocab = vocab;
		this.unknownToken = unknownToken;
		this.maxInputCharactersPerWord = maxInputCharactersPerWord;
	}

    /**
     * A simple whitespace tokenization method that is not used by
     * {@link WordpieceTokenizer}.
     * 
     * @param text the text to tokenize
     * @return
     */
	public static List<String> whitespaceTokenize(String text){
		if(text.isEmpty()) {
			return Collections.emptyList();
		}
		return Arrays.asList(text.split("\\s+"));
	}

    /**
     * Executes Wordpiece tokenization on the provided text after performing
     * whitespace tokenization. This method is not called by
     * {@link WordpieceTokenizer} which calls {@link #wordpiece(String)} directly.
     * Note that tokens corresponding to word suffixes as indicated in the provided
     * vocabulary with the sequence "##" prepended to the entry may be returned by
     * this method. This method does not lowercase the input text or otherwise
     * modify it in any way.
     * 
     * @param text the text to tokenize
     * @return tokens corresponding to Wordpiece tokenization applied to the input
     *         text. Some tokens may have a prefix "##" as described above. Some
     *         tokens may correspond to an unknown token as specified during
     *         initialization (default "[UNK]")
     */
	public List<String> tokenize(String text){
		List<String> outputTokens = new ArrayList<>();

		for(String token : whitespaceTokenize(text)) {
			outputTokens.addAll(wordpiece(token));
		}
		return outputTokens;
	}

    /**
     * Executes Wordpiece tokenization on the provided token. Note that tokens
     * corresponding to word suffixes as indicated in the provided vocabulary with
     * the sequence "##" prepended to the entry may be returned by this method. This
     * method does not perform whitespace tokenization or any other preprocessing.
     * This method does not lowercase the input token or otherwise modify it in any
     * way.
     * 
     * @param token the token to apply Wordpiece tokenization to.
     * @return tokens corresponding to Wordpiece tokenization applied to the input
     *         text. Some tokens may have a prefix "##" as described above. Some
     *         tokens may correspond to an unknown token as specified during
     *         initialization (default "[UNK]")
     */
	public List<String> wordpiece(String token) {
		if(token.length() > this.maxInputCharactersPerWord) {
			return Collections.singletonList(this.unknownToken);
		}

		List<String> subTokens = new ArrayList<>();

		boolean isBad = false;
		int start = 0;
		while(start < token.length()) {
			int end = token.length();
			String currentSubstring = null;
			while(start < end) {
				String substring = token.substring(start, end);
				if(start > 0) {
					substring = "##" + substring;
				}
				if(this.vocab.contains(substring)) {
					currentSubstring = substring;
					break;
				}
				end--;
			}
			if(currentSubstring == null) {
				isBad = true;
				break;
			}
			subTokens.add(currentSubstring);
			start = end;
		}
        if(isBad) {
            return Collections.singletonList(this.unknownToken);
        }
        else {
            return subTokens;
        }
	}

	/**
	 * a getter for the "unknown" token specified during initialization.
	 * @return the "unknown" token name - defaults to "[UNK]"
	 */
	public String getUnknownToken() {
		return unknownToken;
	}

    /**
     * a getter for the maximum character count for a token to consider when
     * {@link #wordpiece(String)} is applied to a token. This value is set at
     * initialization and defaults to 100. Token values passed to that method that
     * are not tokenized and the result of {@link #getUnknownToken()} is returned
     * instead.
     * 
     * @return the maximum length of a token that will be analyzed by
     *         {@link #wordpiece(String)}.
     */
	public int getMaxInputCharactersPerWord() {
		return maxInputCharactersPerWord;
	}

	/**
	 * An OLCUT configurable Wordpiece builder
	 */
    public static final class WordpieceBuilder implements Configurable {
        @Config
        private String vocabPath;
        @Config
        private String unknownToken = "[UNK]";
        @Config
        private int maxInputCharactersPerWord = 100;
        
        public WordpieceBuilder() {}
        
        public WordpieceBuilder(String vocabPath, String unknownToken, int maxInputCharactersPerWord) {
            this.vocabPath = vocabPath;
            this.unknownToken = unknownToken;
            this.maxInputCharactersPerWord = maxInputCharactersPerWord;
        }
        
        public WordpieceBuilder setVocabPath(String vocabPath) {
            this.vocabPath = vocabPath;
            return this;
        }
 
        public WordpieceBuilder setUnknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return this;
        }

        public WordpieceBuilder setMaxInputCharactersPerWord(int maxInputCharactersPerWord) {
            this.maxInputCharactersPerWord = maxInputCharactersPerWord;
            return this;
        }

        public Wordpiece build() throws IOException {
            Set<String> vocab = new HashSet<>(IOUtil.getLines(this.vocabPath));
            return new Wordpiece(vocab, this.unknownToken, this.maxInputCharactersPerWord);
        }
    }

}
