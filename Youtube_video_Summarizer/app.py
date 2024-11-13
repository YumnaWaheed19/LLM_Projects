import re
import streamlit as st 
from css import css
import google.generativeai as genai
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi as yt


def extract_youtube_id(youtube_url):
     video_id = re.findall(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  youtube_url)
     return video_id

def get_transcript(video_id):
        data = yt.get_transcript(video_id[0])
        transcript = ''
        for val in data: 
            for key, value in val.items():
                if key == "text":
                    transcript += value
            l = transcript.splitlines()
            final = " ".join(l)
        return final

def summarize(transcript, model):
    prompt  = [f"You are a YouTube video summarizer. Summarize the video content {transcript} into key points in 200 to 500 words."]
    response =model.generate_content(prompt) 
    responseText = response.text
    if responseText:
        st.title("Summary:")
        text= st.write(responseText)
        return text
    else:
        st.error("Failed to generate summary.")

def main():
        load_dotenv()
        genai.configure(api_key=os.environ['gemini_api_key'])
        generation_config = {
        "temperature": 0.9,
        "max_output_tokens": 1080,
        }

        model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        )

        st.set_page_config(page_title = "YouTube Video Summarizer", page_icon="")
        st.markdown(css, unsafe_allow_html=True)
        st.title("YouTube Video Summarizer")
        st.subheader("Get YouTube transcript and use AI to summarize YouTube videos in just one click for free online with NoteGPT's YouTube summary tool")
        
        placeholder = st.empty()
        youtube_url = st.text_input( "", placeholder="https://www.youtube.com/watch?v=dd1kN_myNDs")
        button = st.button( "Generate Summary")
        if button:
            if youtube_url:
                try:
                    # get video id
                    video_id = extract_youtube_id(youtube_url)
                    # get youtube video transcript
                    placeholder.text("Generating transcript....")
                    transcript = get_transcript(video_id)
                    # Summarize using LLM
                    placeholder.text("Summarizing transcript....")
                    summarize(transcript, model)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                 st.warning("Please add youtube video url")

if __name__ == '__main__':
    main()


