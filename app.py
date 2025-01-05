import os
import base64
from PIL import Image
import google.generativeai as genai
from google.cloud import vision, texttospeech
from flask import Flask, request, jsonify, send_file, render_template
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GOOGLE_API")

# Initialize Gemini API client
 # Replace with your Gemini API key
genai.configure(api_key=api_key)

# Initialize Google Vision and TTS clients
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()

# Flask app setup
app = Flask(__name__)

# Route to display homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_path = "uploaded_image.jpg"
        image.save(image_path)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 400

    with open(image_path, 'rb') as img_file:
        content = img_file.read()

    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)

    if response.error.message:
        return jsonify({"error": f"Vision API error: {response.error.message}"}), 400

    texts = response.text_annotations
    if texts:
        extracted_text = texts[0].description
    else:
        return jsonify({"error": "No text found in the image."}), 400

    # Enhanced prompt for Gemini API
    prompt = (
        f"As a teacher, explain the topic related to this math problem first, using an example if necessary, "
        f"then solve the problem in a way that's easy to understand. The problem is: {extracted_text}"
    )

    try:
        gemini_response = genai.GenerativeModel(model_name="gemini-1.5-pro").generate_content([{
            'mime_type': 'image/jpeg', 'data': base64.b64encode(content).decode('utf-8')}, 
            prompt
        ])
        extracted_problem = gemini_response.text.strip()
    except Exception as e:
        return jsonify({"error": f"Gemini API error: {str(e)}"}), 400

    return jsonify({"solution": extracted_problem})

# Route to handle text-to-speech
@app.route('/speak', methods=['POST'])
def speak_solution():
    solution = request.json.get('solution')
    if not solution:
        return jsonify({"error": "No solution provided"}), 400

    try:
        # Replace '*' (asterisk) with 'multiply' in the solution
        solution = solution.replace('*', '')

        # Use SSML to handle math symbols naturally
        ssml_solution = f"""
        <speak>
            <p>Here is the explanation and solution:</p>
            <p>{solution}</p>
        </speak>
        """

        # Synthesize speech using Google TTS API with enhanced neural voice
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_solution)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-J",  # Advanced neural male voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.05,  # Slightly slower for clarity
            pitch=0,             # Neutral pitch
            volume_gain_db=0.0   # Neutral volume
        )

        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Save the audio to a file
        audio_path = "static/solution.mp3"
        with open(audio_path, "wb") as out:
            out.write(response.audio_content)

        # Send the audio file as a response
        return send_file(audio_path, mimetype='audio/mpeg', as_attachment=True)

    except Exception as e:
        return jsonify({"error": f"TTS API error: {str(e)}"}), 500




if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
