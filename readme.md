#The project runs on python 2.7
Create a virtual environment using the following command virtualenv <virtualenv name> (e.g virtualenv venv)
Activate virtualenvironment using source <virtualenv name>/bin/activate (e.g source venv/bin/activate)
Install all dependencies using pip install -r req.txt (Inside the active virtualenvironment)



Spacy Language installation

#En language model install
 python -m spacy download en



For training the model run the extract_data.py (Open it and uncomment out the following three lines)
For prediction run the prediction.py
