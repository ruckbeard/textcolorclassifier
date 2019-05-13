import pickle

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))


def predict_text_color(red, green, blue):
    return model.predict([[red, green, blue]])

