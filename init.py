import chromadb
import pickle
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

CSV_FILE = 'MovieDatabase.csv'
VECTORIZER_FILE = 'vectorizer.pkl'

def read_movie_data() -> tuple[list, list, list, list]:
    """
    Read movie data from a CSV file.

    Args:
        None

    Returns:
        tuple: A tuple containing the movie data.
    
    Raises:
        None
    """
    ids = []
    documents = []
    titles = []
    metadatas = []

    with open(CSV_FILE, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            row_dict = dict(row)
            
            ids.append(row_dict['id'])
            titles.append(row_dict['Title'])
            metadatas.append({"title": row_dict['Title']})
            
            document = f"{row_dict['Title']} {row_dict['Genre1']} {row_dict['Genre2']} {row_dict['Genre3']} {row_dict['Plot']}"
            documents.append(document)
    
    return ids, documents, titles, metadatas

def create_embeddings(documents: list) -> tuple[list, TfidfVectorizer]:
    """
    Create embeddings for the documents.

    Args:
        documents (list): The list of documents.
    
    Returns:
        list: The list of embeddings.
    
    Raises:
        None
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    embeddings = vectorizer.transform(documents).toarray().tolist()

    return embeddings, vectorizer

def init_db(
    db: chromadb.PersistentClient, 
    collection: chromadb.Collection
    ):
    """
    Initialize the database with movie data.

    Args:
        db (chromadb.PersistentClient): The ChromaDB client.
        collection (chromadb.Collection): The ChromaDB collection.
    
    Returns:
        TfidfVectorizer: The vectorizer used to transform the input.
    
    Raises:
        None
    """
    ids, documents, titles, metadatas = read_movie_data()
    embeddings, vectorizer = create_embeddings(documents)
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)

def init_data():
    db_exists = Path('chroma').exists()
    vectorizer_exists = Path(VECTORIZER_FILE).exists()

    if not db_exists and not vectorizer_exists:
        db = chromadb.PersistentClient()
        collection = db.get_or_create_collection('movies')
        init_db(db, collection)
        print("database and vectorizer initialized.")
    else:
        print("database and vectorizer already exist.")