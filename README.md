## GASLIGHT - Classifier Perturbance
- Use deep reinforcement learning to perturb images in targeted attacks against classifiers.
- For example, Gaslight can attack a number classifier by perturbing pixels in a "7" until it is recognized as a "5"

# To Start
- Head to Gaslight.py and fill out the input parameters as the documentation requires.
- Run Gaslight.py once you have the parameters filled out as desired.

# To tune hyperparameters
- Head to Optuna.py and fill out the input parameters (a subset of values found in Gaslight.py)
- This will create a pkl file of parameters that can be opened up in Gaslight.py.