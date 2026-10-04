from rag_pipeline import load_tokenizer_and_llm, load_data, process_llm_response, process_query, TogetherKeyConfigurationError
from flask import Flask, request, render_template, redirect, url_for, jsonify, g, session
from werkzeug.utils import secure_filename
import os
import textwrap
import time
from extract_image import extract_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = os.urandom(24)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from pdf_extract import convert_pdf_to_text


from datetime import datetime


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


import textwrap

def format_text_into_paragraphs(text):
    # Handle special marker for concluding statement
    parts = text.split("It's important to note")
    main_text = parts[0]
    concluding_text = "It's important to note" + parts[1] if len(parts) > 1 else ""

    # Split the main text by newline characters
    lines = main_text.split('\n')
    
    # Group lines into paragraphs and handle numbered lists
    paragraphs = []
    paragraph = ''
    for line in lines:
        # Check if the line starts with a number followed by a period and space (e.g., "1. ", "10. ")
        if line.strip() and line.strip().split()[0].rstrip('.').isdigit():
            # If paragraph already has content, add it to paragraphs
            if paragraph:
                paragraphs.append(paragraph.strip())
                paragraph = ''
            # Add the numbered line as a new paragraph
            paragraph = line
            paragraphs.append(paragraph.strip())
            paragraph = ''
        else:
            # If not a numbered point, continue adding to the current paragraph
            paragraph += ' ' + line.strip()

    # Add any remaining paragraph to the list
    if paragraph:
        paragraphs.append(paragraph.strip())

    # Add concluding text as a separate paragraph
    if concluding_text:
        paragraphs.append(concluding_text.strip())

    return paragraphs


def clean_extracted_text(text):
    # Split the text into lines
    lines = text.split('\n')

    # Remove leading and trailing spaces from each line
    cleaned_lines = [line.strip() for line in lines]

    # Remove empty lines
    cleaned_lines = [line for line in cleaned_lines if line]

    # Join the lines back into a single string
    cleaned_text = '\n'.join(cleaned_lines)

    return cleaned_text



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = datetime.now()

        query = request.form['query']
        index_mode = request.form.get('index_mode', 'existing')
        try:
            llm = load_tokenizer_and_llm()
        except TogetherKeyConfigurationError as error:
            return render_template('index.html', error=str(error), query=query, index_mode=index_mode), 400
        db = load_data(force_rebuild=index_mode == 'rebuild')
        response = process_query(query, llm, db)

        end_time = datetime.now()
        total_seconds = (end_time - start_time).total_seconds()

        file_name = session.pop('file_name', None)

        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        time_formatted = f"{minutes} minutes, {seconds} seconds"

        return render_template(
            'result.html',
            answer=response['answer'],
            sources=response['sources'],
            filename=file_name,
            time_taken=time_formatted,
            query=query
        )

    return render_template('index.html')


@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['file_name'] = filename

            # Extract text from PDF and save it as 'extracted_text.txt'
            extracted_texts = convert_pdf_to_text(filepath)
            with open('report.txt', 'w') as f:
                for text in extracted_texts:
                    cleaned_text = clean_extracted_text(text)
                    f.write(cleaned_text + "\n")

            # Redirect to main query page after processing
            # return redirect(url_for('index'))
            return jsonify({'status': 'success'})

    return render_template('upload_pdf.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            session['file_name'] = filename

            extract_image(filepath)

            return redirect(url_for('index'))

    return render_template('upload_image.html')



if __name__ == '__main__':
    app.run()
