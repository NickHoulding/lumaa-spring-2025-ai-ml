import chromadb
import pickle
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

CSV_FILE = 'MovieDatabase.csv'
VECTORIZER_FILE = 'vectorizer.pkl'

def get_chroma_db() -> chromadb.PersistentClient:
    """
    Get a reference to the ChromaDB client.

    Args:
        None
    
    Returns:
        chromadb.PersistentClient: A reference to the ChromaDB client.
    
    Raises:
        None
    """
    return chromadb.PersistentClient(path='chroma_db')

def get_top_recs(
    user_input: str, 
    vectorizer: TfidfVectorizer
    ) -> dict:
    """
    Get top recommendations based on user input.

    Args:
        user_input (str): The user input.
        vectorizer (TfidfVectorizer): The vectorizer used to transform the input.

    Returns:
        dict: The top recommendations.
    
    Raises:
        None
    """
    db = get_chroma_db()
    collection = db.get_collection('movies')
    query_vector = vectorizer.transform([user_input]).toarray().tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=5
    )

    return results

def print_recs(recs: dict) -> None:
    """
    Prints the movie recommendations.

    Args:
        recs (dict): The movie recommendations.

    Returns:
        None
    
    Raises:
        None
    """
    print("\nRecommendations:")

    for movie in recs['metadatas'][0]:
        print(movie['title'])

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
    
    return vectorizer

def load_vectorizer() -> TfidfVectorizer:
    """
    Load the saved vectorizer from disk.

    Args:
        None

    Returns:
        TfidfVectorizer: The loaded vectorizer.

    Raises:
        None
    """
    with open(VECTORIZER_FILE, 'rb') as f:
        return pickle.load(f)

def get_input() -> str:
    """
    Gets the user's movie preference input.

    Args:
        None

    Returns:
        str: The user input.

    Raises:
        None
    """
    is_valid = False

    while not is_valid:
        try:
            user_input = str(input("Enter movie preference: ")).strip()
            
            if user_input:
                is_valid = True
            else:
                print("Please enter a valid movie preference.")
        
        except Exception as e:
            print(f"Error: {e}")

    return user_input

def main():
    """Main function to run the program."""
    try:
        db = get_chroma_db()
        
        # Check if database and vectorizer exist
        db_exists = Path('chroma_db').exists()
        vectorizer_exists = Path(VECTORIZER_FILE).exists()

        if not db_exists or not vectorizer_exists:
            collection = db.create_collection('movies')
            vectorizer = init_db(db, collection)
        else:
            vectorizer = load_vectorizer()

        user_input = get_input()
        recs = get_top_recs(user_input, vectorizer)
        print_recs(recs)

    except Exception as e:
        print(f"Error: {e}")

# Entry point
if __name__ == '__main__':
    main()