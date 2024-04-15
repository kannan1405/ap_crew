import os
from dotenv import load_dotenv,find_dotenv
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import requests
from IPython.display import Audio
import streamlit as st

HUGGINGFACE_API_TOKEN=os.getenv("HUGGINGFACE_API_KEY")


# load .env file to environment
load_dotenv(find_dotenv())


# print both variables
def imgtotext(url):

    generate_kwargs = {
        "max_new_tokens": 20,
    }
    captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base",generate_kwargs=generate_kwargs)
    # text=captioner("https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png")
    ## [{'generated_text': 'two birds are standing next to each other '}]
    text=captioner(url)
    return (text[0]['generated_text'])
    


# LLM

def generative_story(scenrio):
    template= """your are story teller .
                you can generate  a short story  based on simple narrative ,the story should be not more than 20 words
                CONTEXT :{scenrio}
                STORY:"""
    prompt=PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    chain=prompt | llm

    story=chain.invoke(scenrio)
    d=dict(story)
    # print(d['content'])
    return d['content']




def main():
    st.set_page_config(page_title="img2story")
    st.header("turn image into story")
    upload_file=st.file_uploader("uplode file ",type="jpg")
    if upload_file is not None:
        print(upload_file)
        bytes_data=upload_file.getvalue()

        with open(upload_file.name,"wb")as file:
            file.write(bytes_data)
        st.image(upload_file,caption="uploaded image.",use_column_width=True)
        scenario=imgtotext(upload_file.name)
        story=generative_story(scenario)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)






if __name__ =='__main__':
    main()

