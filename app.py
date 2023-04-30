import base64
import json

import streamlit as st
import wikipediaapi
import openai


wiki_wiki = wikipediaapi.Wikipedia('en')
openai.api_key = st.secrets["chatgpt-API-key"]

with open('prompts/eli15.txt', 'r') as f:
    explanation_prompt = f.read()

with open('prompts/not_found.txt', 'r') as f:
    not_found_prompt = f.read()


def ask_gpt(prompt):
    response = openai.Completion.create(
        model='text-davinci-002',
        prompt=prompt,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']


def wikipedia_summary(topic):
    page = wiki_wiki.page(topic)

    if not page.exists():
        return page_not_found_handler(topic)

    explanation = ask_gpt(f'{explanation_prompt}\n\nTopic:{topic}\nSummary:{page.summary}')
    return True, explanation, page.fullurl


def page_not_found_handler(topic):
    not_found_response = ask_gpt(f'{not_found_prompt}\n\nTopic: "{topic}"')

    try:
        response_json = json.loads(not_found_response)
    except json.JSONDecodeError:
        return False, f"Something went wrong! Error Code: {base64.b64encode(not_found_response)}", None

    if not response_json['exists']:
        return False, response_json['reason'], None

    page = wiki_wiki.page(response_json['topic'])

    if not page.exists():
        return False, 'Sorry, I don\'t know about this topic...', None

    probable_explanation = ask_gpt(f"{explanation_prompt}{response_json['topic']}\n\n{page.summary}")
    return True, probable_explanation, page.fullurl


def main():
    st.title('ELI15')
    st.write("""
    **Explain Like I\'m 15**: A simple tool to explain complex topics in simple terms.
    I take the topic you provide and search for it on Wikipedia.
    If I find an article on the provided topic, I summarize the it in simple terms.
    If I don\'t find an article on the provided topic, I try to find an article on a similar topic and summarize that.
    """)
    if topic := st.text_input('What do you want to know about?',
                              placeholder='Topic'):
        with st.spinner('Fetching information and summarizing...'):
            success, summary, url = wikipedia_summary(topic)
        if success:
            st.write(summary)
            st.write(f'For more information, visit: {url}')
            st.success('Done!')
        else:
            st.error(summary)


if __name__ == '__main__':
    main()
