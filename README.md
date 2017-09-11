Regression Case Study
======================
The goal of the contest is to predict the sale price of a particular piece of
heavy equipment at auction based on it's usage, equipment type, and
configuration.  The data is sourced from auction result postings and includes
information on usage and equipment configurations.


The evaluation of your model was based on Root Mean Squared Log Error.


Note that this loss function is sensitive to the *ratio* of predicted values to
the actual values, a prediction of 200 for an actual value of 100 contributes
approximately the same amount to the loss as a prediction of 2000 for an actual
value of 1000.
This loss function is implemented in score_model.py.

Setup
======================
Run `pip install git+https://github.com/gschool/dsi-performotron.git`.

Data
======================
The data for this case study are in `./data`.



