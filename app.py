import os
import requests
import streamlit as st
import wikipediaapi
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

wikipedia_base_url = 'https://en.wikipedia.org'
wiki_wiki = wikipediaapi.Wikipedia('en')
os.environ['OPENAI_API_KEY'] = st.secrets["chatgpt-API-key"]
openai.api_key = st.secrets["chatgpt-API-key"]


class Explainer:

    def __init__(self, prompt, model='text-davinci-003', temperature=0.8, max_tokens=2048):
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._prompt_template = PromptTemplate(
            input_variables=['topic', 'wikipedia_research'],
            template=prompt + "\nThe topic is \"{topic}\" and the related Wikipedia Research is as follows."
                              "\n{wikipedia_research}"
        )
        self._langchain_explainer = None

    def _ask(self, query):
        response = openai.Completion.create(
            model=self.model,
            prompt=query,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']

    def explain(self, topic):
        page = wiki_wiki.page(topic)

        if page.exists():
            explanation = self._ask(
                self._prompt_template.format(topic=topic, wikipedia_research=page.summary)
            )
            source = f'Source:\n- {page.fullurl}'
            return explanation, source

        return None, None


class LangchainExplainer:

    def __init__(self, prompt, model='text-davinci-003', temperature=0.8, max_tokens=2048):
        self._llm = OpenAI(model_name=model, temperature=temperature, max_tokens=max_tokens)
        self._prompt_template = PromptTemplate(
            input_variables=['topic', 'wikipedia_research'],
            template=prompt + "\nThe topic is \"{topic}\" and the related Wikipedia Research is as follows."
                              "\n{wikipedia_research}"
        )
        self._llm_chain = LLMChain(
            llm=self._llm,
            prompt=self._prompt_template,
            verbose=True
        )

        self._text_splitter = CharacterTextSplitter()

    def explain(self, topic):
        wiki_loader = WikipediaLoader(query=topic, load_max_docs=5)
        if docs := wiki_loader.load():
            wikipedia_research = "\n".join([
                f"{d.metadata['title']} - {d.metadata['summary']}" for d in docs
            ])

            # Get Explanation
            response = self._llm_chain.run(topic=topic, wikipedia_research=wikipedia_research[:9500])

            return response, wikipedia_research

        return None, None


class ExplanationRefiner:

    def __init__(self, model='text-davinci-003', temperature=0, max_tokens=2048):
        self._llm = OpenAI(model_name=model, temperature=temperature, max_tokens=max_tokens)
        self._text_splitter = CharacterTextSplitter()

    def refine(self, text):
        # Split Explanation into Sentences
        texts = self._text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]

        # Refine Explanation
        refine_chain = load_summarize_chain(self._llm, chain_type="refine")

        return refine_chain.run(docs)


def search_wikipedia(query):
    url = f'{wikipedia_base_url}/w/api.php'
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
        f"{data['query']['pages'][i]['title']} - {wikipedia_base_url}/wiki/?curid={data['query']['pages'][i]['pageid']}"
        for i in data['query']['pages']
    ] if 'query' in data else []


class App:

    def __init__(self, explanation_prompt):
        # Explainers
        self._explainer = Explainer(explanation_prompt)
        self._langchain_explainer = LangchainExplainer(explanation_prompt)
        self._refiner = ExplanationRefiner()

        # UI Elements
        self.content = st.empty()
        self.references = st.empty()
        self.status = st.empty()

    def clear(self):
        self.content.empty()
        self.references.empty()
        self.status.empty()

    def run_explainer(self, topic):
        with st.spinner('Fetching information and summarizing...'):
            summary, source = self._explainer.explain(topic)

        if summary:
            with st.spinner('Refining explanation...'):
                refined_summary = self._refiner.refine(summary)

            self.content.write(refined_summary)
            self.references.write(source)
            return True

        return False

    def run_langchain_explainer(self, topic):
        with st.spinner('Researching...'):
            lc_summary, wikipedia_research = self._langchain_explainer.explain(topic)

        if lc_summary:
            with st.spinner('Refining explanation...'):
                refined_lc_summary = self._refiner.refine(lc_summary)

            self.content.write(refined_lc_summary)

            with self.references.expander('Wikipedia Research'):
                st.write(wikipedia_research)

            return True

        return False

    def not_found_handler(self, topic):
        self.content.error(f'Sorry, the app was not able to find any Wikipedia article on "{topic}".')

        if probable_articles := search_wikipedia(topic):
            articles_str = '\n'.join(f'- {source}' for source in probable_articles)
            self.references.write(f'Were you looking for any of the following articles?\n{articles_str}')

    def run(self, topic):
        self.clear()
        if self.run_explainer(topic):
            self.status.success('Done!')
        elif self.run_langchain_explainer(topic):
            self.status = st.container()
            self.status.info(f'The topic "{topic}" does not directly correspond to any Wikipedia article.')
            self.status.warning('The app did its own research and came up with the above explanation.')
            self.status.info('You can find the Wikipedia research performed by the app in the expander above.')
        else:
            self.not_found_handler(topic)


def main():
    st.title('ELI15')
    st.write("""
    **Explain Like I\'m 15**: A simple tool to explain complex topics in simple terms.
    I take the topic you provide and search for it on Wikipedia.
    If I find an article on the provided topic, I summarize it in simple terms.
    If I don\'t find an article on the provided topic, I do my own research and summarize it in simple terms.
    """)
    tp = st.empty()

    # Load Explanation Prompt
    with open('prompts/eli15.txt', 'r') as f:
        explanation_prompt = f.read()

    # Initialize App
    app = App(explanation_prompt)

    if topic := tp.text_input('Just provide the topic or subject you want to know about!',
                              placeholder='Topic'):
        app.run(topic)


if __name__ == '__main__':
    main()
