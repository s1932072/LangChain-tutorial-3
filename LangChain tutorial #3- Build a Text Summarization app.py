import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    # Instantiate the LLM model
    llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    # Split text
    text_splitter = CharacterTextSplitter(
        #chunk_size = 50,  # チャンクの文字数
        #chunk_overlap = 10,  # チャンクオーバーラップの文字数
    )
    texts = text_splitter.split_text(txt)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# Page title
st.set_page_config(page_title='🦜🔗 Text Summarization App')
st.title('🦜🔗 Text Summarization App')

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Form to accept user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    #openai_api_key = st.text_input('OpenAI API Key', type = 'password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')
    if submitted :
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result.append(response)
            #del openai_api_key
if len(result):
    st.info(response)