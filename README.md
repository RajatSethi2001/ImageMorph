## GASLIGHT - Classifier Perturbance
- Use deep reinforcement learning to perturb objects in attacks against classifiers.
- Gaslight works against almost any numpy array, as long as minor changes aren't noticeable and there's a min/max for each value in the array.
- For example, Gaslight can attack a number classifier by perturbing pixels in a "7" until it is recognized as a "5"
- Gaslight can also perform untargeted attacks, where it perturbs until the label is different from the original.
- Gaslight works best with a score for each label (like Softmax), but it can also work with hard-label tasks, though those take much longer.

# To Start
- Head to Gaslight.py and fill out the input parameters as the documentation requires.
- Run Gaslight.py once you have the parameters filled out as desired.

# To tune hyperparameters
- Head to Optuna.py and fill out the input parameters (a subset of values found in Gaslight.py)
- This will create a pkl file of parameters that can be opened up in Gaslight.py.