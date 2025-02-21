# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation
## Dataset
- This movie dataset is from HuggingFace's datasets library:  
https://huggingface.co/datasets/moizmoizmoizmoiz/MovieRatingDB.
- It is included in this repository, and is 250 rows in size. The data will be initialized automatically the first time `recommend.py` is run.

## Setup
- Python3 version: `3.10.12`
- Create Virtual environment: `python3 -m venv venv`
- Start the virtual environment: 
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `.\venv\Scripts\activate`
- Install Dependencies: `pip install -r requirements.txt`

## Running
- Run the code: `python recommend.py`
- Enter your movie preference when prompted by the program.

## Results
Example:  
Enter movie preference: Thrilling action movies set in space.  
  
Movie Recommendations:  
- Aliens  
- Mad Max: Fury Road  
- The Iron Giant  
- Taxi Driver  
- Rush