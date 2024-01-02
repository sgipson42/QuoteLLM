import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# choose a file
# choose a row

filename = "/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/quotes-results.csv"
df = pd.read_csv(filename)
file = filename.split("/")[-1]
genre = file.split("-results.csv")[0]  # "quotes"

title = genre.split("-")
spaced_title = " ".join(title)
caps_title = spaced_title.title() # "Quotes"
#suing-works
    # row number: 8
# change row number to graph different quote
row_num = 199
tensor_scores = df.iloc[row_num]["cosine_scores"]
optimal_score = df.iloc[row_num]["optimal_cosine"]

# get optimal score as a float
score = optimal_score.split("tensor([[")
score_split = score[1].split("]])")
score_split = score_split[:len(score_split)-1]
str_score = score_split[0]
opt_score = float(str_score)
print("Opt", opt_score)
# get list of scores as a float list
str_scores = tensor_scores.split(" ")
scores = []
for score in str_scores:
    score = score.split("tensor([[")
    score_split = score[1].split("]])")
    score_split = score_split[:len(score_split)-1]
    str_score = score_split[0]
    score = float(str_score)
    print(score)
    scores.append(score)

print(scores)
print(len(scores))
print(opt_score)
x = np.arange(1, len(scores)+1)
print(x)


max_val = max(scores)
max_index = scores.index(max_val)+1

y = scores
plt.xlabel("Sentence Length")
plt.ylabel("Cosine Score")
plt.title(f"Cosine Score per Sentence Length for {caps_title}")
plt.plot(x, y, marker = "o")  # Plot the chart
# plt.plot(max_index, max_val, color='red', label='Max Value')
# plt.annotate(f'Max Value: {max_val}', (max_index, max_val), xytext=(max_index + 0.5, max_val + 0.5),
             #arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
plt.savefig(f"/Users/skyler/Desktop/QuoteLLM/{genre}_cosine_scores_{row_num}.png")
# 8 is index in dataframe not index in excel -- index in excel is 10
plt.show()  # display
