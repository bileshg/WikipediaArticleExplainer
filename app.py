import json
import re
import requests
import streamlit as st
import wikipediaapi
import openai
import spacy_streamlit

wiki_wiki = wikipediaapi.Wikipedia('en')
spacy_model = "en_core_web_sm"
openai.api_key = st.secrets["chatgpt-API-key"]

with open('prompts/eli15.txt', 'r') as f:
    explanation_prompt = f.read()


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


def search_wikipedia(query):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'origin': '*',
        'format': 'json',
        'generator': 'search',
        'gsrnamespace': 0,
        'gsrlimit': '5',
        'gsrsearch': query
    }

    data = requests.get(url, params=params).json()

    return [
        f"{data['query']['pages'][i]['title']} - http://en.wikipedia.org/wiki/?curid={data['query']['pages'][i]['pageid']}"
        for i in data['query']['pages']
    ]


def not_found_handler(topic):
    articles = search_wikipedia(topic)

    return (
        False,
        'Sorry, unable to determine the topic against the query...',
        articles
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
            articles_str = '\n'.join(f'- {source}' for source in sources)
            st.write(f'Were you looking for any of the following articles?\n{articles_str}')


if __name__ == '__main__':
    main()
