# %%
import phe as paillier
import json
import numpy as np

# %%
def getData():
    with open('./Data/data.json', 'r') as file: 
        d=json.load(file)
    data=json.loads(d)
    return data
getData()

# %%
def getCoef():
    with open('./Data/Coefficients.json', 'r') as file: 
        d=json.load(file) 
        coef = np.array(d)
    return coef   
getCoef()

# %%
def get_Intercept():
    with open('./Data/Intercept.json', 'r') as file: 
        d=json.load(file)
    return d   
    

# %%
def get_encryted_intercept():
    data = getData()
    intercept = get_Intercept()
    public_key=paillier.PaillierPublicKey(n=int(data['public_key']['n']))
    encrypted_intercept = public_key.encrypt(intercept)
    return encrypted_intercept 

# %%
def computeData():
    data=getData()
    mycoef=getCoef()
    intercept = get_encryted_intercept()
    pk = data['public_key']
    pubkey= paillier.PaillierPublicKey(n=int(pk['n']))
    print(pubkey)
    enc_nums_rec = [paillier.EncryptedNumber(pubkey, int(x[0]), int(x[1])) for x in data['values']]
    results=(sum([mycoef[i]*enc_nums_rec[i] for i in range(len(mycoef))])+ intercept)
    return results, pubkey

# %%
computeData()

# %%
def serializeData():
    results, pubkey = computeData()
    encrypted_data={}
    encrypted_data['pubkey'] = {'n': pubkey.n}
    encrypted_data['values'] = (str(results.ciphertext()), results.exponent)
    serialized = json.dumps(encrypted_data)
    return serialized


# %%
def main():
    datafile=serializeData()
    with open('./Data/answer.json', 'w') as file:
        json.dump(datafile, file)

# %%
main()

# %%



