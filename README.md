# processing_text

The provided data .csv must have the following columns:
"name", "text", "synonyms"

"name": the label name. Can be empty but the column must exist
"text": the general data. MUST be populated
"synonyms": comma-seperated synonyms of the label in one cell. Can be empty or just populated with the label name instead

You must define a context(as in name) of the corpus, the path to the data, and type of data(used for hashing)
You might also have to download a few Python libraries. Here's the list:

numpy

pandas

nltk.stem

gensim