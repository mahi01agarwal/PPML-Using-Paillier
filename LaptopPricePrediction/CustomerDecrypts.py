# %%
import phe as paillier
import json
import numpy as np


# %%

def getKeys():

    with open('./Data/custkeys.json', 'r') as file: 
        keys=json.load(file)
        pub_key=paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
        priv_key=paillier.PaillierPrivateKey(pub_key,keys['private_key']['p'],keys['private_key']['q'])
        return pub_key, priv_key 

# %%
pub_key, priv_key = getKeys()

# %%
def loadAnswer():
    with open('./Data/answer.json', 'r') as file: 
        ans=json.load(file)
    answer=json.loads(ans)
    return answer

# %%
answer_file=loadAnswer()
answer_key=paillier.PaillierPublicKey(n=int(answer_file['pubkey']['n']))
answer = paillier.EncryptedNumber(answer_key, int(answer_file['values'][0]), int(answer_file['values'][1]))
if (answer_key==pub_key):
    pred_ans = (priv_key.decrypt(answer))
    final_pred = np.exp(pred_ans)
    print(final_pred)

# %%



