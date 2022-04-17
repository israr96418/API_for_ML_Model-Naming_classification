import joblib
from fastapi import FastAPI,Query

# The Main idea for this API development to classify b/w name
# we have two catagories one Male and other one is Female:
# when you enter your name our ML model classify whether it is male or female name:
# Befor going to Api we import ML package


app = FastAPI()


# vectorization means vectorize name:
gender_vectorize = open('models/gender_vectorizer.pkl',"rb")
gender_cv = joblib.load(gender_vectorize)
print("Gender Vectorization: ",gender_cv)

gender_model = open('models/gender_nv_model.pkl',"rb")
gender_clf = joblib.load(gender_model)
print("Gender Model: ",gender_clf)

@app.get("/")
def Get_data():
    return {"Hello Api Lover"}


@app.get("/item/{name}")
def Get_name(name: str):
    return {"Name: ": name}


# ML API:
@app.get("/prediction/{name}")
async  def prediction(name:str):
	vectorize_name=gender_cv.transform([name]).toarray()
	prediction = gender_clf.predict(vectorize_name)
	if prediction[0]==0:
		result = "female"
	else:
		result = "male"
	return {"origin_name":name, "prediction":result}


