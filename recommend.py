import chromadb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from init import init_data
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
    return chromadb.PersistentClient(path='chroma')

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
    print("\nMovie Recommendations:")

    for movie in recs['metadatas'][0]:
        print("- " + movie['title'])

def get_vectorizer() -> TfidfVectorizer:
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
        db_exists = Path('chroma').exists()
        vectorizer_exists = Path(VECTORIZER_FILE).exists()

        if not db_exists and not vectorizer_exists:
            print("Initializing data...")
            init_data()

        db = get_chroma_db()
        vectorizer = get_vectorizer()
        user_input = get_input()

        recs = get_top_recs(user_input, vectorizer)
        print_recs(recs)

    except Exception as e:
        print(f"Error: {e}")

# Entry point
if __name__ == '__main__':
    main()