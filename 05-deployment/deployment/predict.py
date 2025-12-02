import pickle

# Load the model
model_file = 'pipeline_v1.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in) #Unpacking the pipeline

# Validation Data
customer = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dv.transform([customer])

lead_score = model.predict_proba(X)[0, 1]

print(lead_score)

# Alternatively (Using the pipeline directly without unpacking it)

# import pickle

# with open('pipeline_v1.bin', 'rb') as f_in:
#     pipeline = pickle.load(f_in)

# customer = {
#     "lead_source": "paid_ads",
#     "number_of_courses_viewed": 2,
#     "annual_income": 79276.0
# }

# lead_score = pipeline.predict_proba([customer])[0, 1]
# print(f"Lead score: {lead_score:.3f}")

"""
Pipeline directly without unpacking it explained
The Pipeline object is a single scikit-learn estimator that contains both the 
DictVectorizer (which does one-hot encoding) and the classifier. 

When you call pipeline.predict_proba(...), the pipeline automatically runs the 
data through all of its steps in order – first the vectorizer, then the model – 
so the raw dictionary is turned into a one-hot-encoded 2D numeric matrix before 
the logistic regression ever sees it.
"""

