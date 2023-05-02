import base64
import json
import re
import streamlit as st
import wikipediaapi
import openai
import spacy_streamlit

wiki_wiki = wikipediaapi.Wikipedia('en')
spacy_model = "en_core_web_sm"
openai.api_key = st.secrets["chatgpt-API-key"]

with open('prompts/eli15.txt', 'r') as f:
    explanation_prompt = f.read()

with open('prompts/not_found.txt', 'r') as f:
    not_found_prompt = f.read()


def ask_gpt(prompt, model='text-davinci-002', temperature=0.7):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def get_explanation(wikipedia_page):
    explanation = ask_gpt(
        f'{explanation_prompt}\n\nTopic:{wikipedia_page.title}\nSummary:{wikipedia_page.summary}',
        model='text-davinci-003',
        temperature=0.8
    )
    return True, explanation, [wikipedia_page.fullurl]


def wikipedia_summary(topic):
    page = wiki_wiki.page(topic)

    return (
        get_explanation(page)
        if page.exists()
        else process_as_sentence(topic)
    )


def is_question(doc):
    return next(
        (
            token.tag_ == "VBZ" and token.text.endswith("?")
            for token in doc
            if token.dep_ == "ROOT"
        ),
        False,
    )


def get_nouns(doc):
    return [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]


def remove_articles(text):
    return re.sub(r'^(a|an|the)\s+', '', text)


def process_as_sentence(text):
    nlp = spacy_streamlit.load_model(spacy_model)
    doc = nlp(text)

    summaries = []
    sources = []
    for chunk in doc.noun_chunks:
        if chunk.ents:
            st.info(f'Processing chunk: **{chunk.text}**')
            page = wiki_wiki.page(remove_articles(chunk.text))
            if page.exists():
                summaries.append(page.summary)
                sources.append(page.fullurl)

    if not summaries:
        return not_found_handler(text)

    prompt = "\n\n".join(summaries) + "\n\n" + text
    return True, ask_gpt(prompt), sources


def not_found_handler(topic):
    st.warning("This could take a while...")

    not_found_response = ask_gpt(f'{not_found_prompt} \"{topic}\"', temperature=0)

    try:
        response_json = json.loads(not_found_response)
    except json.JSONDecodeError:
        return False, f"Something went wrong! Error Code: {base64.b64encode(not_found_response)}", None

    if not response_json['exists']:
        return False, response_json['reason'], None

    page = wiki_wiki.page(response_json['topic'])
    return (
        get_explanation(page)
        if page.exists()
        else (
            False,
            'Sorry, I was unable to find a Wikipedia article on this subject...',
            None,
        )
    )


def main():
    st.title('ELI15')
    st.write("""
    **Explain Like I\'m 15**: A simple tool to explain complex topics in simple terms.
    I take the topic you provide and search for it on Wikipedia.
    If I find an article on the provided topic, I summarize it in simple terms.
    If I don\'t find an article on the provided topic, I try to find an article on a similar topic and summarize that.
    """)
    if topic := st.text_input('Just provide the topic or subject you want to know about!',
                              placeholder='Topic'):
        with st.spinner('Fetching information and summarizing...'):
            success, summary, sources = wikipedia_summary(topic)
        if success:
            st.write(summary)
            sources_str = '\n'.join(f'- {source}' for source in sources)
            st.write(f'Sources:\n{sources_str}')
            st.success('Done!')
        else:
            st.error(summary)


if __name__ == '__main__':
    main()
