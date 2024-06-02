import os
import uuid
from flask import Flask, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import speech_recognition as sr
from deep_translator import GoogleTranslator
from google.transliteration import transliterate_text
import torch
from TTS.api import TTS
from df.enhance import enhance, init_df, load_audio, save_audio


app = Flask(__name__)
app.config["STORAGE_FOLDER"] = "storage"
os.makedirs(app.config["STORAGE_FOLDER"], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

df_model, df_state, _ = init_df()

recognizer = sr.Recognizer()

language_codes = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-CN": "Chinese (Simplified)",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "es": "Spanish",
}


@app.route("/translate", methods=["POST"])
def translate():
    input_lang = request.form.get("input_lang", "ja")
    output_lang = request.form.get("output_lang", "en")

    if input_lang not in language_codes or output_lang not in language_codes:
        return jsonify({"error": "Invalid language code"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["STORAGE_FOLDER"], filename)
        file.save(file_path)

        # deep filter
        audio, _ = load_audio(file_path, sr=df_state.sr())
        enhanced = enhance(df_model, df_state, audio)
        enhanced_output_path = os.path.join(
            app.config["STORAGE_FOLDER"], f"enhanced_{str(uuid.uuid4())}.wav"
        )
        save_audio(enhanced_output_path, enhanced, df_state.sr())

        try:
            with sr.AudioFile(file_path) as source:
                audio = recognizer.record(source)
                speech_text = recognizer.recognize_google(audio, language=input_lang)
                speech_text_transliteration = (
                    transliterate_text(speech_text, lang_code=input_lang)
                    if input_lang not in ("auto", "en")
                    else speech_text
                )
                translated_text = GoogleTranslator(
                    source=input_lang, target=output_lang
                ).translate(text=speech_text_transliteration)

                # TTS
                clone_output_filename = f"cloned_{str(uuid.uuid4())}.mp3"
                clone_output_path = os.path.join(
                    app.config["STORAGE_FOLDER"], clone_output_filename
                )
                tts = TTS(TTS_MODEL).to(DEVICE)
                tts.tts_to_file(
                    text=translated_text,
                    speaker_wav=enhanced_output_path,
                    language="en",
                    file_path=clone_output_path,
                )

                response = {
                    "recognized_text": speech_text_transliteration,
                    "translated_text": translated_text,
                    "audio_file_url": url_for(
                        "download_file", filename=clone_output_filename, _external=True
                    ),
                }

                return jsonify(response)
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand the audio!"}), 400
        except sr.RequestError:
            return jsonify({"error": "Could not request results from Google!"}), 500
        finally:
            os.remove(file_path)


@app.route("/uploads/<filename>")
def download_file(filename):
    return send_file(
        os.path.join(app.config["STORAGE_FOLDER"], filename), as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
