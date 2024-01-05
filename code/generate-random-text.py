import tiktoken
import numpy as np

token_enc = tiktoken.get_encoding("cl100k_base")

for i in range (10):
    rand_token_ids = np.random.randint(0, token_enc.max_token_value+1, size=(1000,))
    rand_txt = token_enc.decode(rand_token_ids)
    # print(rand_txt)
    f = open(f"/Users/skyler/oldLLM/transcripts/random_text/random_{i}.txt", "w")
    f.write(rand_txt)
    f.close()
