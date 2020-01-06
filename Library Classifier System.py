# necessary imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def preprocessing(dataframe1, dataframe2):
    # drop all rows with no subject
    dataframe.dropna(subset = ['Subjects'], how='all', inplace=True, axis=0) 

    Y = dataframe['MaterialType']
    X = dataframe['Subjects']

    # one hot encoding of classes i.e. MaterialType
    encoder = LabelBinarizer()
    transfomed_label = encoder.fit_transform(Y)

    # appending train and test data to process them together
    combi = X.append(test_data['Subjects'], ignore_index=True)
    
    # converting into numpy
    combi['tidy'] = np.vectorize(X)
    
    # replace any special characters with a white space
    combi['tidy'] = combi['tidy'].str.replace("[^a-zA-Z#]", " ")

    # remove words that are too short
    combi['tidy'] = combi['tidy'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>3]))
    
    return combi['tidy'], Y, encoder

def bag_of_words(column):
    # bow_vectorizer
    bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
    bow = bow_vectorizer.fit_transform(column)

    #separating out train and test
    train_bow = bow[:29890,:]
    test_bow = bow[29890:,:]
    
    return train_bow, test_bow

def training_and_submission(train_values, test_values, targ, enc, df):
    #splitting training data and cross validation data in 80:20
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_values, targ, random_state = 42, test_size=0.2)

    #model and fitting
    lreg = LogisticRegression()
    lreg.fit(xtrain_bow, ytrain)

    # predicting on the validation set
    prediction = lreg.predict_proba(xvalid_bow)
    prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 then 1 else 0
    prediction_int = prediction_int.astype(np.int)
    enc.inverse_transform(prediction)
    print('Score on cross validation:', lreg.score(xvalid_bow, yvalid))

    # test data prediction and submission to csv
    test_pred = lreg.predict_proba(test_values)
    test_pred == test_pred.max(axis = 1)[:, None].astype(np.int)
    enp = enc.inverse_transform(test_pred)
    enp = enp[:len(enp)-1]
    df['MaterialType'] = enp
    subm = df[['ID','MaterialType']]
    subm.to_csv('sub_lreg.csv', index=False)

if __name__ == '__main__':
    train_data = pd.read_csv(r'C:\Users\Hp\Downloads\train_file.csv')
    test_data = pd.read_csv(r'C:\Users\Hp\Downloads\test_file.csv')
    col, target, encoding = preprocessing(train_data, test_data)
    train_val, test_val = bag_of_words(col)
    training_and_submission(train_val, test_val, target, test_data)
    
    
