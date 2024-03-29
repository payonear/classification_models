# netguru-case

## About
This project is a solution of Netguru programming task.

## Structure
- `model.py` - Loads or creates a classification model, that as an input takes vectorized text embeddings and predicts classes.
- `models`
- - `net.py` - Neural network for classification based on text embeddings.
- `utils`
- - `dataset.py` - Pytorch Dataset object created for comfortable interaction with torch DataLoaders.
- `Solution.ipynb` - Jupyter Notebook with description of learning process for 4 predictive models.

## Running
- You need to download the data from [here](https://drive.google.com/file/d/1pzQOxzzqPBBzdTwRYxR8IJ1KEWgpnFw_/view)
- Unzip the data to `data` folder in project directory
- create virtual environment `python -m venv venv` and activate it (optionally)
- run in terminal `pip install -r requirements.txt`
- if using venv run `python -m ipykernel install --user --name=myenv` with name of your venv
- open `Solution.ipynb`
- if using venv choose repective kernel
