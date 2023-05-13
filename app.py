import os
import streamlit as st
import wikipediaapi
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.utilities import WikipediaAPIWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

wiki_wiki = wikipediaapi.Wikipedia('en')
os.environ['OPENAI_API_KEY'] = st.secrets["chatgpt-API-key"]
openai.api_key = st.secrets["chatgpt-API-key"]


class Explainer:

    def __init__(self, prompt, model='text-davinci-003', temperature=0.8):
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self._prompt_template = PromptTemplate(
            input_variables=['topic', 'wikipedia_research'],
            template=prompt + "\nThe topic is \"{topic}\" and the related Wikipedia Research is as follows."
                              "\n{wikipedia_research}"
        )
        self._langchain_explainer = None

    def _ask(self, prompt):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=2048,
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

        self._wiki = WikipediaAPIWrapper()
        self._text_splitter = CharacterTextSplitter()

    def explain(self, topic):
        # Wikipedia Research
        wikipedia_research = self._wiki.run(topic)

        # Get Explanation
        response = self._llm_chain.run(topic=topic, wikipedia_research=wikipedia_research)

        return response, wikipedia_research


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


def main():
    st.title('ELI15')
    st.write("""
    **Explain Like I\'m 15**: A simple tool to explain complex topics in simple terms.
    I take the topic you provide and search for it on Wikipedia.
    If I find an article on the provided topic, I summarize it in simple terms.
    If I don\'t find an article on the provided topic, I try to find an article on a similar topic and summarize that.
    """)

    with open('prompts/eli15.txt', 'r') as f:
        explanation_prompt = f.read()
    explainer = Explainer(explanation_prompt)
    langchain_explainer = LangchainExplainer(explanation_prompt)
    refiner = ExplanationRefiner()

    if topic := st.text_input('Just provide the topic or subject you want to know about!',
                              placeholder='Topic'):
        content = st.empty()
        references = st.empty()
        with st.spinner('Fetching information...'):
            summary, source = explainer.explain(topic)

        if summary:
            with st.spinner('Summarizing...'):
                refined_summary = refiner.refine(summary)

            content.write(refined_summary)
            references.write(source)
        else:
            with st.spinner('Researching...'):
                lc_summary, wikipedia_research = langchain_explainer.explain(topic)

            with st.spinner('Refining explanation...'):
                refined_lc_summary = refiner.refine(lc_summary)

            content.write(refined_lc_summary)

            with references.expander('Wikipedia Research'):
                st.write(wikipedia_research)


if __name__ == '__main__':
    main()
