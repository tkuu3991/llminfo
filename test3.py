from flask import Flask, request, jsonify, send_file  , render_template  
from flask_cors import CORS  
import azure.cognitiveservices.speech as speechsdk  
from openai import AzureOpenAI  
from dotenv import load_dotenv  
import os  
from PIL import Image  
import io  
import requests  
import base64  
from azure.cognitiveservices.vision.computervision import ComputerVisionClient  
from msrest.authentication import CognitiveServicesCredentials  
from pydub import AudioSegment  
import subprocess
from google.cloud import texttospeech
import logging
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
import time
from azure.storage.blob import BlobServiceClient

logging.basicConfig(level=logging.DEBUG)  
app = Flask(__name__)  
CORS(app)  
  
load_dotenv()  

audio_path = r"C:\Users\chqi1\OneDrive\Desktop\week2\output\output.wav"

# Azure Speech Service  
speech_key = '3bd2cecbf3fe49d58c1abf3a33a58027'  
service_region = 'eastus'  
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)  
speech_config.speech_recognition_language = "zh-HK"
speech_config.speech_synthesis_voice_name = "zh-HK-HiuGaaiNeural"  # Add this line
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)  
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)  
  
# Azure OpenAI Service  
client = AzureOpenAI(  
    api_key='defb3cb0b60840e29d145d74fdc9b8ec',  
    api_version="2024-02-01"  
)  
deployment = 'gpt-4o'  
  
# Azure Computer Vision Service  
subscription_key = "fb69606823b54dc8b484359539ac7fa6"  
endpoint = "https://llmvisioncwh.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))  


# Initialize Blob Service Client
connect_str = 'DefaultEndpointsProtocol=https;AccountName=storagecwh;AccountKey=1Av/Y6L7F94AkTH1Z/AI8SLhMMBlp5wyIL3f8Ipf0sZ4HHd1Yk7N47E9TOeh52hzf7BJCYx8lLUA+ASt895puw==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'cwhllm'



@app.route('/')
def home():
    return render_template('test1.html')
  
def upload_to_blob(file_path, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded {file_path} to {container_name}/{blob_name}")

def complete_AzureOpenAI(prompt):  
    try:  
        system_message = {  
            "role": "system",  
            "content": "You are an expert teacher. Respond professionally to learning-related questions."  
        }  
        user_message = {  
            "role": "user",  
            "content": prompt  
        } 

        response = client.chat.completions.create(  
            model="gpt-4o",  
            messages=[system_message, user_message],  # Include system message
            temperature=0.7,  # Adjust temperature for more focused responses
            max_tokens=500,  
            top_p=1,  
            presence_penalty=0,  
            frequency_penalty=0,  
            stop=None  
        )  

        response_text = response.choices[0].message.content.strip()
        return response_text

    except Exception as e:  
        print("An exception occurred:", e)  
        return None
  
@app.route('/recognize', methods=['POST'])  
def recognize():  
    print("Recognition request received")  
    result = speech_recognizer.recognize_once_async().get()  
  
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:  
        print("Recognition result: {}".format(result.text))  
        response = complete_AzureOpenAI(result.text)  
        print("Chatbot response: {}".format(response))  
        return jsonify({'recognition': result.text, 'response': response})  
  
    elif result.reason == speechsdk.ResultReason.NoMatch:  
        print("No speech could be recognized.")  
        return jsonify({'error': 'No speech could be recognized.'})  
  
    elif result.reason == speechsdk.ResultReason.Canceled:  
        cancellation_details = result.cancellation_details  
        print("Recognition canceled: {}".format(cancellation_details.reason))  
        if cancellation_details.reason == speechsdk.CancellationReason.Error:  
            print("Error: {}".format(cancellation_details.error_details))  
        return jsonify({'error': 'Recognition canceled: {}'.format(cancellation_details.reason)})  
  
@app.route('/message', methods=['POST'])  
def message():  
    data = request.get_json()  
    message = data.get('message')  
    if message:  
        response = complete_AzureOpenAI(message)  
        return jsonify({'response': response})  
    else:  
        return jsonify({'error': 'No message provided'})  
  
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        app.logger.error("No image file provided")
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']
    if image.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(image)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # OCR for text extraction
        read_response = computervision_client.read_in_stream(buffer, language='zh-Hans', raw=True)
        operation_location = read_response.headers.get("Operation-Location")
        if not operation_location:
            app.logger.error("Failed to get Operation-Location")
            return jsonify({'error': 'Failed to process image'}), 500

        operation_id = operation_location.split("/")[-1]

        # Wait for OCR operation to complete
        while True:
            result = computervision_client.get_read_result(operation_id)
            if result.status not in [OperationStatusCodes.not_started, OperationStatusCodes.running]:
                break
            time.sleep(1)

        # Reset buffer for image analysis
        buffer.seek(0)

        # Image analysis for description
        analysis = computervision_client.analyze_image_in_stream(buffer, visual_features=['Description'])

        description = analysis.description.captions[0].text if analysis.description.captions else "No description available"

        app.logger.debug("Image description: %s", description)

        # Synthesize description to speech
        audio_config = speechsdk.audio.AudioOutputConfig(filename=os.path.join(output_folder, "output.wav"))
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        speech_result = synthesizer.speak_text_async(description).get()

        if speech_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            app.logger.debug("Synthesis completed successfully")
            # Convert to MP3
            wav_path = os.path.join(output_folder, "output.wav")
            mp3_path = os.path.join(output_folder, "output.mp3")
            ffmpeg_path = r"C:\Program Files\FFMPEG\bin\ffmpeg"
            command = [ffmpeg_path, "-y", "-i", wav_path, mp3_path]
            subprocess.run(command, check=True)
            return jsonify({
                'description': description,
                'message': 'Audio synthesis completed',
                'audio_url': '/audio'
            })
        else:
            app.logger.error("Synthesis failed")
            return jsonify({'error': 'Failed to synthesize audio'}), 500

    except Exception as e:
        app.logger.error("An exception occurred: %s", str(e))
        return jsonify({'error': 'Failed to process image'}), 500
 

output_folder = r"C:\Users\chqi1\OneDrive\Desktop\week2\output"  
if not os.path.exists(output_folder):  
    os.makedirs(output_folder)  
  
@app.route('/synthesize', methods=['POST'])  
def synthesize():  
    data = request.get_json()  
    text = data.get('text')  
    if text:  
        try:  
            wav_path = os.path.join(output_folder, "output.wav")  
            mp3_path = os.path.join(output_folder, "output.mp3")  
            audio_config = speechsdk.audio.AudioOutputConfig(filename=wav_path)  
  
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)  
            result = synthesizer.speak_text_async(text).get()  
  
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:  
                app.logger.debug("Synthesis completed successfully")  
                if os.path.exists(wav_path):  
                    try:  
                        ffmpeg_path = r"C:\Program Files\FFMPEG\bin\ffmpeg"  
                        command = [ffmpeg_path, "-y", "-i", wav_path, mp3_path]  
                        subprocess.run(command, check=True)  
  
                        if os.path.exists(mp3_path):  
                            # Upload MP3 to Azure Blob Storage
                            upload_to_blob(mp3_path, 'output.mp3')
                            return jsonify({'message': 'Synthesis completed', 'audio_url': '/audio'})  
                        else:  
                            return jsonify({'error': 'MP3 file not found after synthesis'}), 500  
                    except Exception as e:  
                        app.logger.error(f"Error converting WAV to MP3: {e}")  
                        return jsonify({'error': f"Error converting WAV to MP3: {e}"}), 500  
                else:  
                    app.logger.error("WAV file not found after synthesis")  
                    return jsonify({'error': 'WAV file not found after synthesis'}), 500  
            elif result.reason == speechsdk.ResultReason.Canceled:  
                cancellation_details = result.cancellation_details  
                app.logger.error(f"Synthesis canceled: {cancellation_details.reason}")  
                return jsonify({'error': 'Synthesis canceled: {}'.format(cancellation_details.reason)})  
        except Exception as e:  
            app.logger.error(f"Exception during synthesis: {e}")  
            return jsonify({'error': str(e)}), 500  
    else:  
        return jsonify({'error': 'No text provided'})  

  
@app.route('/play_audio', methods=['GET'])  
def play_audio():  
    mp3_path = os.path.join(output_folder, "output.mp3")  
    if os.path.exists(mp3_path):  
        return render_template('play_audio.html')  
    else:  
        return jsonify({'error': 'Audio file not found'}), 404  
  
@app.route('/audio', methods=['GET'])  
def get_audio():  
    try:  
        mp3_path = os.path.join(output_folder, "output.mp3")  
        if os.path.exists(mp3_path):  
            return send_file(mp3_path, mimetype='audio/mp3')  
        else:  
            return jsonify({'error': 'Audio file not found'}), 404  
    except Exception as e:  
        return jsonify({'error': str(e)}), 500  
  



if __name__ == '__main__':  
    app.run(debug=True)  

  




