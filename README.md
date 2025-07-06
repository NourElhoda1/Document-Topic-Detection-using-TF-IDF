# Document Topic Detection using TF-IDF & LSA

This project is a lightweight **Flask-based web application** that analyzes `.txt` or `.pdf` documents and detects their **dominant topics**, using two different Natural Language Processing (NLP) techniques:

- **TF-IDF (Term Frequency - Inverse Document Frequency)** 
- **LSA (Latent Semantic Analysis)** 

---

##  Objectives

- Provide users with a simple interface to upload text or PDF files.
- Automatically identify the main **topics** within the document.
- Compare two approaches for topic extraction: **TF-IDF** and **LSA**.

---

##  Features

-  Upload support for `.txt` and `.pdf` files
-  Text cleaning and preprocessing
-  Topic detection using TF-IDF + keyword dictionary
-  Semantic analysis using LSA (Truncated SVD)
-  Simple web interface built with Flask
-  Toggle between TF-IDF and LSA analysis modes

---

##  Project Structure

```
index_projet/
│
├── app.py                    # Main Flask application
├── theme_dictionary.py      # Dictionary of keywords by topic (TF-IDF)
├── templates/
│   └── index.html           # Web interface
├── static/
│   └── style.css            # Styling
└── uploads/                 # Temporary uploaded files
```

---

##  Technologies Used

| Type         | Tools & Libraries                         |
|--------------|-------------------------------------------|
| Language     | Python 3                                  |
| Framework    | Flask                                     |
| NLP Tools    | scikit-learn (TF-IDF, TruncatedSVD)       |
| PDF Parsing  | PyPDF2                                    |
| Frontend     | HTML5, CSS3                               |

---

##  How to Run the App Locally

1. **Clone the repository**
```bash
git clone https://github.com/NourElhoda1/Document-Topic-Detection-using-TF-IDF.git
cd Document-Topic-Detection-using-TF-IDF---LSA
```

2. **(Optional) Create a virtual environment**
```bash
python -m venv env
env\Scripts\activate     # On Windows
```

3. **Install the dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the Flask app**
```bash
python app.py
```

5. Open your browser at:
```
http://127.0.0.1:5000/
```




