import streamlit as st
import wikipediaapi
import openai


wiki_wiki = wikipediaapi.Wikipedia('en')
openai.api_key = st.secrets["chatgpt-API-key"]

with open('prompts/eli15.txt', 'r') as f:
    explanation_prompt = f.read()


def wikipedia_summary(topic):
    page = wiki_wiki.page(topic)
    if page.exists():
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=f'{explanation_prompt}\n\nTopic:{topic}\nSummary:{page.summary}',
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return True, response['choices'][0]['text'], page.fullurl
    else:
        return False, 'Sorry, I don\'t know anything about that topic...', None


def main():
    st.title('ELI15')
    st.write('**Explain Like I\'m 15**: A simple tool to explain complex topics in simple terms')
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
