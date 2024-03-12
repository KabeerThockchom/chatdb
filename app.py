#app.py
from dotenv import load_dotenv
load_dotenv()

from functools import wraps
from flask import Flask, jsonify, Response, request, redirect, url_for
import flask
import os
from cache import MemoryCache

app = Flask(__name__, static_url_path='')

# SETUP
cache = MemoryCache()

api_key = os.getenv("OPENAI_API_KEY")

from vanna.local import LocalContext_OpenAI
vn = LocalContext_OpenAI(config={'api_key':api_key, 'model': 'gpt-4'})

vn.connect_to_sqlite('/Users/kabeerthockchom/Downloads/fifa24.sqlite')

# # ONLY NEED TO DO ONCE
# df_information_schema = vn.run_sql("""SELECT type, sql FROM sqlite_master WHERE sql is not null""") 
# df_information_schema
# for ddl in df_information_schema['sql'].to_list():
#   vn.train(ddl=ddl)
import os
import whisper
from flask import jsonify
import datetime
import sounddevice as sd
import wavio as wv

# Directory to store recordings
recordings_dir = 'recordings'
if not os.path.exists(recordings_dir):
    os.makedirs(recordings_dir)

# Whisper model
model = whisper.load_model("base")

# File to store transcriptions
TRANSCRIPT_FILE = 'transcript.txt'

# Audio recording parameters
freq = 44100  # Sample rate
duration = 5  # Duration in seconds



def record_audio():
    print('Recording...')
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H-%M-%S.wav")
    filepath = os.path.join(recordings_dir, filename)

    # Record audio
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()  # Wait until recording is finished

    # Save recorded audio
    wv.write(filepath, recording, freq, sampwidth=2)
    print(f"Saved recording to {filepath}")

    # Transcribe the audio file
    transcribe_audio(filepath)

def transcribe_audio(filepath):
    print(f"Transcribing {filepath}...")
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language='en', fp16=False)
    result = model.transcribe(audio)

    if result['text']:
        print(f"Transcription: {result['text']}")
        # Append text to transcript file
        with open(TRANSCRIPT_FILE, 'a') as f:
            f.write(result['text'] + '\n')

        # Return the transcribed text as a response
        return jsonify({"text": result['text']})
    else:
        print("No speech detected")
        return jsonify({"type": "error", "error": "No speech detected"})
    
@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y-%m-%d_%H-%M-%S.wav")
    filepath = os.path.join(recordings_dir, filename)
    
    # Record audio
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()  # Wait until recording is finished
    
    # Save recorded audio
    wv.write(filepath, recording, freq, sampwidth=2)
    print(f"Saved recording to {filepath}")
    
    response = transcribe_audio(filepath)
    return response
# NO NEED TO CHANGE ANYTHING BELOW THIS LINE
def requires_cache(fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            id = request.args.get('id')

            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})
            
            for field in fields:
                if cache.get(id=id, field=field) is None:
                    return jsonify({"type": "error", "error": f"No {field} found"})
            
            field_values = {field: cache.get(id=id, field=field) for field in fields}
            
            # Add the id to the field_values
            field_values['id'] = id

            return f(*args, **field_values, **kwargs)
        return decorated
    return decorator

@app.route('/api/v0/generate_questions', methods=['GET'])
def generate_questions():
    return jsonify({
        "type": "question_list", 
        "questions": vn.generate_questions(),
        "header": "Here are some questions you can ask:"
        })

@app.route('/api/v0/generate_sql', methods=['GET'])
def generate_sql():
    question = flask.request.args.get('question')

    if question is None:
        return jsonify({"type": "error", "error": "No question provided"})

    id = cache.generate_id(question=question)
    sql = vn.generate_sql(question=question)

    cache.set(id=id, field='question', value=question)
    cache.set(id=id, field='sql', value=sql)

    return jsonify(
        {
            "type": "sql", 
            "id": id,
            "text": sql,
        })

@app.route('/api/v0/run_sql', methods=['GET'])
@requires_cache(['sql'])
def run_sql(id: str, sql: str):
    try:
        df = vn.run_sql(sql=sql)

        cache.set(id=id, field='df', value=df)

        return jsonify(
            {
                "type": "df", 
                "id": id,
                "df": df.head(10).to_json(orient='records'),
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})

# @app.route('/api/v0/run_sql', methods=['GET'])
# @requires_cache(['sql'])
# def run_sql(id: str, sql: str):
#     try:
#         df = vn.run_sql(sql=sql)
#         df_json = df.head(10).to_dict(orient='records')
#         cache.set(id=id, field='df', value=df_json)
#         return jsonify({
#             "type": "df",
#             "id": id,
#             "df": df_json
#         })
#     except Exception as e:
#         return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/download_csv', methods=['GET'])
@requires_cache(['df'])
def download_csv(id: str, df):
    csv = df.to_csv()

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 f"attachment; filename={id}.csv"})

@app.route('/api/v0/generate_plotly_figure', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_plotly_figure(id: str, df, question, sql):
    try:
        code = vn.generate_plotly_code(question=question, sql=sql, df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
        fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
        fig_json = fig.to_json()

        cache.set(id=id, field='fig_json', value=fig_json)

        return jsonify(
            {
                "type": "plotly_figure", 
                "id": id,
                "fig": fig_json,
            })
    except Exception as e:
        # Print the stack trace
        import traceback
        traceback.print_exc()

        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/get_training_data', methods=['GET'])
def get_training_data():
    df = vn.get_training_data()

    return jsonify(
    {
        "type": "df", 
        "id": "training_data",
        "df": df.head(25).to_json(orient='records'),
    })

@app.route('/api/v0/remove_training_data', methods=['POST'])
def remove_training_data():
    # Get id from the JSON body
    id = flask.request.json.get('id')

    if id is None:
        return jsonify({"type": "error", "error": "No id provided"})

    if vn.remove_training_data(id=id):
        return jsonify({"success": True})
    else:
        return jsonify({"type": "error", "error": "Couldn't remove training data"})

@app.route('/api/v0/train', methods=['POST'])
def add_training_data():
    question = flask.request.json.get('question')
    sql = flask.request.json.get('sql')
    ddl = flask.request.json.get('ddl')
    documentation = flask.request.json.get('documentation')

    try:
        id = vn.train(question=question, sql=sql, ddl=ddl, documentation=documentation)

        return jsonify({"id": id})
    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/generate_followup_questions', methods=['GET'])
@requires_cache(['df', 'question', 'sql'])
def generate_followup_questions(id: str, df, question, sql):
    followup_questions = vn.generate_followup_questions(question=question, sql=sql, df=df)

    cache.set(id=id, field='followup_questions', value=followup_questions)

    return jsonify(
        {
            "type": "question_list", 
            "id": id,
            "questions": followup_questions,
            "header": "Here are some followup questions you can ask:"
        })

@app.route('/api/v0/load_question', methods=['GET'])
@requires_cache(['question', 'sql', 'df', 'fig_json', 'followup_questions'])
def load_question(id: str, question, sql, df, fig_json, followup_questions):
    try:
        return jsonify(
            {
                "type": "question_cache", 
                "id": id,
                "question": question,
                "sql": sql,
                "df": df.head(10).to_json(orient='records'),
                "fig": fig_json,
                "followup_questions": followup_questions,
            })

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})

@app.route('/api/v0/get_question_history', methods=['GET'])
def get_question_history():
    return jsonify({"type": "question_history", "questions": cache.get_all(field_list=['question']) })

@app.route('/')
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
