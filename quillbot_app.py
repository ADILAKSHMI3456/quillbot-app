import streamlit as st
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from transformers import pipeline
import difflib
import random
from collections import defaultdict
import ssl

# ============== INITIAL SETUP ==============
st.set_page_config(page_title="QuillBot-like Tool", layout="wide")

# ============== NLTK DATA DOWNLOAD ==============
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

required_nltk_data = ['punkt', 'averaged_perceptron_tagger', 'brown', 'wordnet', 'conll2000']

for dataset in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else 
                      f'taggers/{dataset}' if dataset == 'averaged_perceptron_tagger' else 
                      f'corpora/{dataset}')
    except LookupError:
        nltk.download(dataset)

# ============== MODEL LOADING ==============
@st.cache_resource
def load_models():
    try:
        paraphraser = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return paraphraser, summarizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

paraphraser, summarizer = load_models()

# ============== CORE FUNCTIONS ==============
def get_synonyms(word, pos_tag=None):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    
    if word in synonyms:
        synonyms.remove(word)
    
    if pos_tag:
        pos_mapping = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'J': wordnet.ADJ,
            'R': wordnet.ADV
        }
        simplified_pos = pos_tag[0].upper()
        if simplified_pos in pos_mapping:
            pos_synonyms = set()
            for syn in wordnet.synsets(word, pos=pos_mapping[simplified_pos]):
                for lemma in syn.lemmas():
                    pos_synonyms.add(lemma.name().replace('_', ' '))
            if pos_synonyms:
                return list(pos_synonyms)
    
    return list(synonyms) if synonyms else []

def simple_paraphrase(text, level=2):
    words = word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    new_words = []
    
    for word, tag in pos_tags:
        synonyms = get_synonyms(word, tag)
        if synonyms and random.random() < (level * 0.1):
            new_words.append(random.choice(synonyms))
        else:
            new_words.append(word)
    
    return ' '.join(new_words)

def advanced_paraphrase(text):
    if not paraphraser:
        return "Paraphraser model not loaded."
    try:
        result = paraphraser(text, max_length=len(text.split()) * 3, do_sample=True, top_k=50)
        return result[0]['generated_text']
    except Exception as e:
        return f"Error in paraphrasing: {str(e)}"

def summarize_text(text, max_length=130, min_length=30):
    if not summarizer:
        return "Summarizer model not loaded."
    try:
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error in summarization: {str(e)}"

def check_grammar(text):
    try:
        blob = TextBlob(text)
        corrections = []
        
        for sentence in blob.sentences:
            corrected = str(sentence.correct())
            if corrected != str(sentence):
                corrections.append((str(sentence), corrected))
        
        return corrections
    except Exception as e:
        st.error(f"Grammar check failed. Please ensure NLTK data is installed.")
        return []

# ============== STREAMLIT UI ==============
def main():
    st.title("✨ QuillBot-like Writing Assistant")
    st.markdown("Enhance your writing with paraphrasing, grammar checking, summarization, and more!")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        paraphrase_level = st.select_slider(
            "Paraphrase Intensity",
            options=["Standard", "Fluency", "Creative"],
            value="Standard"
        )
        output_mode = st.radio(
            "Output Display",
            options=["Side by Side", "Inline Changes"],
            index=0
        )
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("This tool provides QuillBot-like features including:")
        st.markdown("- ✍️ Paraphrasing at different levels")
        st.markdown("- 📝 Grammar checking")
        st.markdown("- 📊 Summarization")
        st.markdown("- 🔍 Synonym suggestions")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Paraphrase", "Grammar Check", "Summarize", "Synonyms"])

    with tab1:
        st.subheader("Paraphrase Your Text")
        input_text = st.text_area("Enter text to paraphrase:", height=150, 
                                placeholder="Type or paste your text here...", key="paraphrase_input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Text**")
            st.write(input_text if input_text else "Enter some text to see the original version here.")
        
        with col2:
            st.markdown("**Paraphrased Text**")
            if input_text:
                if paraphrase_level == "Standard":
                    paraphrased = simple_paraphrase(input_text, level=2)
                elif paraphrase_level == "Fluency":
                    paraphrased = simple_paraphrase(input_text, level=1)
                else:
                    paraphrased = advanced_paraphrase(input_text)
                
                if output_mode == "Side by Side":
                    st.write(paraphrased)
                else:
                    diff = difflib.ndiff(input_text.split(), paraphrased.split())
                    diff_text = []
                    for word in diff:
                        if word.startswith('+ '):
                            diff_text.append(f"<span style='color:green'>{word[2:]}</span>")
                        elif word.startswith('- '):
                            diff_text.append(f"<span style='color:red'>{word[2:]}</span>")
                        elif word.startswith('  '):
                            diff_text.append(word[2:])
                    st.markdown(' '.join(diff_text), unsafe_allow_html=True)
                
                st.markdown(f"**Similarity:** {difflib.SequenceMatcher(None, input_text, paraphrased).ratio()*100:.1f}%")
            else:
                st.info("The paraphrased version will appear here.")

    with tab2:
        st.subheader("Grammar Checker")
        grammar_text = st.text_area("Enter text to check:", height=150, 
                                  placeholder="Type or paste your text here...", key="grammar_input")
        
        if grammar_text:
            with st.spinner("Checking grammar..."):
                corrections = check_grammar(grammar_text)
                if corrections:
                    st.warning(f"Found {len(corrections)} potential issue(s):")
                    for original, corrected in corrections:
                        st.markdown(f"- **Original:** {original}")
                        st.markdown(f"  **Suggested:** {corrected}")
                        st.markdown("---")
                else:
                    st.success("No grammar issues found!")
        else:
            st.info("Enter some text to check for grammar issues.")

    with tab3:
        st.subheader("Text Summarizer")
        summarize_text_input = st.text_area("Enter text to summarize:", height=200, 
                                          placeholder="Paste a long article or document here...", key="summary_input")
        
        if summarize_text_input:
            with st.spinner("Generating summary..."):
                summary = summarize_text(summarize_text_input)
                st.markdown("**Summary**")
                st.write(summary)
                
                original_length = len(word_tokenize(summarize_text_input))
                summary_length = len(word_tokenize(summary))
                compression_ratio = (1 - (summary_length / original_length)) * 100
                st.markdown(f"Reduced from {original_length} to {summary_length} words ({compression_ratio:.1f}% compression)")
        else:
            st.info("Enter some text to generate a summary.")

    with tab4:
        st.subheader("Synonym Finder")
        synonym_word = st.text_input("Enter a word to find synonyms:", 
                                   placeholder="Type a word...", key="synonym_input")
        
        if synonym_word:
            words = word_tokenize(synonym_word)
            if len(words) > 1:
                st.warning("Please enter a single word for synonym lookup.")
            else:
                pos_tags = nltk.pos_tag(words)
                synonyms = get_synonyms(words[0], pos_tags[0][1] if pos_tags else None)
                
                if synonyms:
                    st.markdown(f"**Synonyms for '{words[0]}':**")
                    pos_groups = defaultdict(list)
                    for syn in synonyms:
                        syn_pos = None
                        for synset in wordnet.synsets(syn):
                            if synset.pos():
                                syn_pos = synset.pos()
                                break
                        
                        pos_name = {
                            'n': 'Noun',
                            'v': 'Verb',
                            'a': 'Adjective',
                            's': 'Adjective Satellite',
                            'r': 'Adverb'
                        }.get(syn_pos, 'Other')
                        
                        pos_groups[pos_name].append(syn)
                    
                    for pos, syns in pos_groups.items():
                        with st.expander(f"{pos} ({len(syns)})"):
                            st.write(", ".join(syns))
                else:
                    st.info(f"No synonyms found for '{words[0]}'")

if __name__ == "__main__":
    main()
