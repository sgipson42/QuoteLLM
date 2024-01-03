import tiktoken
import numpy as np
token_enc = tiktoken.get_encoding("cl100k_base")
rand_token_ids = np.random.randint(0, token_enc.max_token_value+1, size=(1000,))
rand_txt = token_enc.decode(rand_token_ids)
# print(rand_txt)
tokens_for_text = token_enc.encode("Decomposition")
print(tokens_for_text)
rand_txt = token_enc.decode(tokens_for_text)
print(rand_txt)