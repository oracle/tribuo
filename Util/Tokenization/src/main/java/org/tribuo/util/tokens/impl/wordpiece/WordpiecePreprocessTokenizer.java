package org.tribuo.util.tokens.impl.wordpiece;

import org.tribuo.util.tokens.impl.SplitFunctionTokenizer;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 *  
 */
public class WordpiecePreprocessTokenizer extends SplitFunctionTokenizer {

    public static SplitFunction createSplitFunctionTokenizer(boolean tokenizeChineseChars) {

        return (codepoint, index, cs) -> {
            if (Character.isWhitespace(codepoint)) {
                return SplitResult.SPLIT_AT;
            }
            if (codepoint == 160) { // \u00a0 (NO-BREAK SPACE)
                return SplitResult.SPLIT_AT;
            }
            if (isPunctuation(codepoint)) {
                return SplitResult.SPLIT_BEFORE_AND_AFTER_PUNCTUATION;
            }
            if (tokenizeChineseChars && isChinese(codepoint)) {
                return SplitResult.SPLIT_BEFORE_AND_AFTER_WORD;
            }

            if (codepoint == 0 || codepoint == 0xFFFD || isControl(codepoint)) {
                return SplitResult.SPLIT_AT;
            }

//          int charType = Character.getType(codepoint);
            // if(charType == Character.OTHER_LETTER) { //charType ==
            // Character.COMBINING_SPACING_MARK ||
//              return SplitType.SPLIT_BEFORE_AND_AFTER;
//          }
            return SplitResult.NO_SPLIT_WORD;
        };

    }

    public static boolean isPunctuation(int codepoint) {
        if (codepoint >= 33 && codepoint <= 47) {
            return true;
        }
        if (codepoint >= 58 && codepoint <= 64) {
            return true;
        }
        if (codepoint >= 91 && codepoint <= 96) {
            return true;
        }
        if (codepoint >= 123 && codepoint <= 126) {
            return true;
        }

        int charType = Character.getType(codepoint);
        if (charType == Character.DASH_PUNCTUATION || charType == Character.START_PUNCTUATION
                || charType == Character.END_PUNCTUATION || charType == Character.CONNECTOR_PUNCTUATION
                || charType == Character.OTHER_PUNCTUATION || charType == Character.INITIAL_QUOTE_PUNCTUATION
                || charType == Character.FINAL_QUOTE_PUNCTUATION) {
            return true;
        }

        return false;
    }

    public static boolean isChinese(int codepoint) {
        if ((codepoint >= 0x4E00 && codepoint <= 0x9FFF) || (codepoint >= 0x3400 && codepoint <= 0x4DBF)
                || (codepoint >= 0x20000 && codepoint <= 0x2A6DF) || (codepoint >= 0x2A700 && codepoint <= 0x2B73F)
                || (codepoint >= 0x2B740 && codepoint <= 0x2B81F) || (codepoint >= 0x2B820 && codepoint <= 0x2CEAF)
                || (codepoint >= 0xF900 && codepoint <= 0xFAFF) || (codepoint >= 0x2F800 && codepoint <= 0x2FA1F)) {
            return true;
        }
        return false;
    }

    public static boolean isControl(int codepoint) {
        char c = Character.toChars(codepoint)[0];

        // this is a soft-hyphen that isn't caught as a format character in the python
        // implementation. You can experiment with adding these lines back in if you are
        // getting
        // hyphen related regressions between the two tokenizers
//      if(c == '­') {
//          return false;
//      }

        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        int charType = Character.getType(codepoint);
        if (charType == Character.CONTROL || charType == Character.FORMAT || charType == Character.PRIVATE_USE
                || charType == Character.SURROGATE) {
            return true;
        }
        return false;
    }

//    if char == "\t" or char == "\n" or char == "\r":
//        return False
//    cat = unicodedata.category(char)
//    if cat.startswith("C"):
//        return True
//    return False

    @Config(description = "split on Chinese tokens?")
    private boolean tokenizeChineseChars = true;

    public WordpiecePreprocessTokenizer() {
        this.postConfig();
    }

    public WordpiecePreprocessTokenizer(boolean tokenizeChineseChars) {
        this.tokenizeChineseChars = tokenizeChineseChars;
        this.postConfig();
    }

    @Override
    public void postConfig() {
        this.setSplitFunction(createSplitFunctionTokenizer(this.tokenizeChineseChars));
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public WordpiecePreprocessTokenizer clone() {
        return new WordpiecePreprocessTokenizer();
    }

}
