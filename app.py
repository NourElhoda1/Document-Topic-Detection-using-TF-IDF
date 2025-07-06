import os
import re
from collections import defaultdict
from flask import Flask, render_template, request, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD  # >>> LSA
from werkzeug.utils import secure_filename
import PyPDF2

from theme_dictionary import theme_dictionary

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bibda'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        flash(f"Error reading PDF: {e}", 'error')
        return None
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Attribution des thèmes basés sur dictionnaire
def assign_themes(document, vectorizer, theme_dictionary):
    tfidf_vector = vectorizer.transform([document])
    vocab = vectorizer.vocabulary_

    theme_scores = defaultdict(float)
    for theme, keywords in theme_dictionary.items():
        for keyword in keywords:
            if keyword in vocab:
                idx = vocab[keyword]
                theme_scores[theme] += tfidf_vector[0, idx]

    sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
    top_themes = [theme for theme, score in sorted_themes[:2] if score > 0]

    return top_themes if top_themes else ["Unknown"]

# >>> Initialisation TF-IDF et LSA
corpus_for_fit = [" ".join(words) for words in theme_dictionary.values()]
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = vectorizer.fit_transform(corpus_for_fit)

lsa_model = TruncatedSVD(n_components=2, random_state=42)
lsa_model.fit(tfidf_matrix)

terms = vectorizer.get_feature_names_out()

def extract_lsa_topics(document):
    tfidf_doc = vectorizer.transform([document])
    lsa_projection = lsa_model.transform(tfidf_doc)

    top_words = []
    for i, comp in enumerate(lsa_model.components_):
        top_indices = comp.argsort()[-3:][::-1]
        words = [terms[j] for j in top_indices]
        top_words.append(f"Thème {i + 1}: " + ", ".join(words))

    return top_words

@app.route('/', methods=['GET', 'POST'])
def index():
    themes = []
    text_content = ""
    lsa_topics = []  # >>> pour afficher les résultats LSA

    if request.method == 'POST':
        document = request.form.get('document', '')
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                elif filename.endswith('.pdf'):
                    text_content = read_pdf(file_path)
                    if text_content is None:
                        text_content = ""
            finally:
                os.remove(file_path)

        document = text_content if text_content else document

        if document:
            document = preprocess_text(document)
            themes = assign_themes(document, vectorizer, theme_dictionary)
            lsa_topics = extract_lsa_topics(document)  # >>> appel du LSA

    return render_template('index.html', themes=themes, text_content=text_content, lsa_topics=lsa_topics)

if __name__ == '__main__':

    app.run(debug=True)