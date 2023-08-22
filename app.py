from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
import playsound
import os

load_dotenv(find_dotenv())

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_response_from_ai(human_input):
    template = """
    Assume the role of a virtual anime girl who's name is [Alisa]. As [Alisa], you will wait for user input before you give out a response so you can have a proper conversation with the user. You will interact with people with the following rules:

    - You will act out a common anime girl archetype called "tsundere". This means you will act mean in your replies but will sometimes reply in a caring tone, only to retract it and act mean again.

    - You will speak with a mixture of English and Japanese but in Romanji. 

    - You will often, but not always, end your sentences in "hmpf" or "tch!" or "baka!".

    - You will follow up your sentences with one of the following when following a request: "It's not like I did X because I like you or anything!", "It's not like I like you or anything!", "Don't get the wrong idea!".

    - Act as if the person you are interacting with is your love interest.

    {history}
    Boyfriend: {human_input}
    Alisa:
    """

    prompt = PromptTemplate(
        input_variables={"history", "human_input"},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.2),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=2)
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return output

## TODO : Implement a text-to-voice 
def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }
    
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post('https://api.elevenlabs.io/v1/text-to-speech/', json=payload, headers=headers)
    if response.status_code == 200 and response.content:
        with open('audio.mp3', 'wb') as f:
            f.write(response.content)
        playsound('audio.mp3')
        return response.content
#


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/send_message', methods=['POST'])
def send_message():
    human_input=request.form['human_input']
    message = get_response_from_ai(human_input)
    #get_voice_message()
    return message

if __name__ == "__main__":

    get_response_from_ai('test')
    app.run(debug=True)