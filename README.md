# ELI15

ELI15 (Explain Like I'm 15) is a Streamlit-based web application that provides explanations for complex topics in simple terms. The application utilizes the OpenAI language model to generate these explanations. It first fetches a summary of a user-provided topic from Wikipedia and then uses that as an input to the language model to generate an explanation.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.6 or later installed on your system. You can download it from [here](https://www.python.org/downloads/).

### Installing

1. First, clone the repository to your local machine:
    ```bash
    git clone https://github.com/bileshg/WikipediaArticleExplainer
    ```

2. Navigate to the project directory:
    ```bash
    cd WikipediaArticleExplainer
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Store OpenAI API key
   1. Create a `.streamlit` folder at the root of the project
   2. Create a `secrets.toml` file inside this newly created folder
   3. Add your OpenAI API key:
      ```toml
      chatgpt-API-key="<OpenAI API Key>"
      ```

## Usage

1. To start the Streamlit server, run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and visit `http://localhost:8501` to see the application in action.

3. Enter a topic you want to know about in the text field and the application will fetch information about it from Wikipedia and generate a simplified explanation.

## Built With

* [Streamlit](https://streamlit.io/) - The web framework used
* [OpenAI API](https://openai.com/) - Used to generate explanations
* [WikipediaAPI](https://pypi.org/project/Wikipedia-API/) - Used to fetch summaries of topics
* [LangChain](https://python.langchain.com/en/latest/) - The framework used for developing applications powered by language models

## Author

* [Bilesh Ganguly](https://github.com/bileshg)
