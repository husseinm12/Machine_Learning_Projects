This project is a content-based movie recommendation system. It suggests movies that are similar to a user’s favorite movie based on features like genres, keywords, tagline, cast, and director.

📌How It Works ?

1- Import Libraries
*numpy and pandas → for data manipulation.
*difflib → to match user input with the closest movie title in the dataset.
*scikit-learn → for text vectorization (TfidfVectorizer) and similarity calculation (cosine_similarity).

2- Load Dataset
*The system loads movies.csv into a Pandas DataFrame.
*Example columns: title, genres, keywords, tagline, cast, director.

3- Feature Selection & Preprocessing
*Relevant features: genres, keywords, tagline, cast, director.
*Missing values (NaN) are replaced with empty strings.
*All selected features are combined into a single text field per movie.

4- Text Vectorization
*TfidfVectorizer converts textual features into numerical vectors.
*Each movie is represented as a vector in a multi-dimensional space.

5- Similarity Calculation
cosine_similarity compares the movie vectors.
The similarity score measures how close two movies are.

6- User Input
*The user enters a favorite movie name.
*difflib.get_close_matches finds the closest matching title in the dataset (handles typos and variations).

7- Recommendation Generation
*The system finds the index of the selected movie.
*Retrieves similarity scores for all movies compared to this movie.
*Sorts them in descending order of similarity.
*Displays the top 30 recommended movies.

🛠️ Requirements

Install dependencies with:
pip install -r requirements.txt

▶️ Usage

Place movies.csv in the same folder as the script.
Run the script: python movie_recommender.py
Enter your favorite movie name when prompted.
The system prints recommended movies.

🚀 Example Workflow

Input: "Iron Man"
Output: A ranked list of movies like "Iron Man 2", "The Avengers", "Spider-Man: Homecoming", etc.




