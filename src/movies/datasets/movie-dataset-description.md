# K.E.T.S. PROJECT Movie Dataset Description

## Dataset Overview
This dataset to contain information about movies, including ratings, genres, and other metadata.

## Data Columns
The dataset consists of 10 columns with 29,697 rows.

1. **TCONST** (object):
   - Unique IMDB identifier for each entry
   - Corresponds to IMDb's tconst (title constant) identifier

2. **ORIGINAL_TITLE** (object):
   - The original title of the movie or TV show

3. **TYPE** (object):
   - Indicates whether the entry is a movie, TV show, or other media type

4. **AVG_RATING** (float64):
   - The average rating of the title
   - Numeric values, on a scale (e.g., 1-10)

5. **VOTE_COUNT** (int64):
   - The number of votes or ratings received
   - Integer values

6. **GENRES** (object):
   - The genre(s) associated with the title
   - Likely a string, possibly comma-separated if multiple genres

7. **DIRECTORS** (object):
   - The director(s) of the title
   - Ccomma-separated if multiple directors

8. **YEAR** (object):
   - The year of release or production
   - Stored as an object (string) rather than a numeric type

9. **OVERVIEW** (object):
   - Brief synopsis or description of the title

10. **COMBINED_FEATURES** (object):
    - A column that combines multiple features (ORIGINAL_TITLE, GENRES, YEAR, DIRECTORS, OVERVIEW) for TFIDF processing
    - The exact composition of this column is not specified in the provided information

## Data Types
- 8 columns are of type 'object' (likely strings)
- 1 column is of type 'float64' (AVG_RATING)
- 1 column is of type 'int64' (VOTE_COUNT)

