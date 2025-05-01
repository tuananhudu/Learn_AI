from sklearn.preprocessing import MinMaxScaler

def normalize(X , scaler = None):
    if scaler is None : 
        scaler=  MinMaxScaler()
        scaler.fit(X)

    return scaler.transform(X) , scaler , 
