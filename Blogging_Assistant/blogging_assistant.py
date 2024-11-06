import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import pdfkit 
import os

load_dotenv()
genai.configure(api_key=os.environ['gemini_api_key'])

generation_config = {
  "temperature": 0.9,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 2048,
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

st.set_page_config(layout='wide')
title = "Blogging Assistant: Your AI Writing Tool 🤖 "
st.markdown(f"<h1 style='font-size: 30px;'>{title}</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Your Blog Details")
    blog_titles = st.text_input("Blog Tiltle")
    keywords = st.text_area("Keywords [comma-seperated]")
    number_of_words = st.slider("Number of words", min_value = 250, max_value = 1000 , step = 100)
    option = st.selectbox(
        'Select Writing tone',
        ("Standard", "Professional", "Formal", "Academic", "Fluent")
    )
    prompt = [f"Generate a comprehensive, engaging blog post relevant to the given title \"{blog_titles}\" and keywords \"{keywords}\". Make sure to incorporate these words in the blog post. The blog should be approximately {number_of_words} words in length, suitable for an online audience. Ensure the content is original, informative and maintains a consistant tone {option} throughout\n"]
    button = "Generate Blog"
    generate_blog_button = st.button(button)


if  generate_blog_button :

    if not blog_titles or not keywords:
        st.error("Please provide both a blog title and keywords.")
    else:
        response =model.generate_content(prompt) 
        responseText = response.text
        st.write(responseText)
        blog_html = f""" <html>
        <head>
            <title>{blog_titles}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                h1 {{ color: black; }}
                p {{ font-size: 14px; }}
            </style>
        </head>
        <body>
            <h1>{blog_titles}</h1>
            <p>{responseText}</p>
        </body>
        </html>
        """

        html_filename = blog_titles+".html"
        with open(html_filename, "w") as html_file:
            html_file.write(blog_html)

        # Convert HTML to PDF using pdfkit
        pdf_filename = blog_titles + ".pdf"
        path_to_wkhtmltopdf = r'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe' 
        config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)
        pdfkit.from_file(html_filename, pdf_filename, configuration=config)

        with open(pdf_filename, "rb") as f:
            PDFbyte = f.read()

        # Download button for the generated PDF
        st.download_button(
            label="Download Blog as PDF",
            data=PDFbyte,
            file_name=blog_titles + ".pdf",
            mime='application/pdf'
        )