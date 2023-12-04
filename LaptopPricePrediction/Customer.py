# %%
import phe as paillier
import json

def storeKeys():

    public_key, private_key = paillier.generate_paillier_keypair()
    keys={}
    keys['public_key'] = {'n': public_key.n}
    print(public_key)
    keys['private_key'] = {'p': private_key.p,'q':private_key.q}
    with open('./Data/custkeys.json', 'w') as file: 
        json.dump(keys, file)
        
        
storeKeys()


# %%


# %%
def getKeys():
    with open('./Data/custkeys.json', 'r') as file: 
        keys=json.load(file)
        pub_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
        priv_key=paillier.PaillierPrivateKey(pub_key,keys['private_key']['p'],keys['private_key']['q'])
        return pub_key, priv_key 

# %%
def serializeData(public_key, data):
    encrypted_data_list = [public_key.encrypt(x) for x in data]
    encrypted_data={}
    encrypted_data['public_key'] = {'n': public_key.n}
    encrypted_data['values'] = [(str(x.ciphertext()), x.exponent) for x in encrypted_data_list]
    serialized = json.dumps(encrypted_data)
    return serialized

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
# Load the col_tnf object from the file
with open('col_tnf.pkl', 'rb') as file:
    col_tnf = pickle.load(file)
    
col_tnf

# %%


# %%
pub_key, priv_key = getKeys()
#Company	TypeName	Ram	OpSys	Weight	os	Cpu brand	ProcessorSpeed	Gpu brand	GpuModel	HDD	SSD	Hybrid	Flash_Storage	TouchScreen	IPSPanel	PPI
data = [['Dell', 'Notebook',8,'Windows 10',2.00,'Windows','Intel Core i7',2.8,'Nvidia',1050.0,0,256,0,0,0,0,141.0211998]]
trans_data = col_tnf.transform(data)

new_data = trans_data.reshape(-1)
new_data


# %%
serializeData(pub_key, new_data)
datafile = serializeData(pub_key, new_data)
with open('./Data/data.json', 'w') as file: 
    json.dump(datafile, file)

# %%


# %%


# %%



