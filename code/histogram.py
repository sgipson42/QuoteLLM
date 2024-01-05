import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/Users/skyler/Desktop/QuoteLLM/results3.0/CSVs/"
csv_file = csv_path + "constitution-results.csv"
graph_title = "Constitution"
graph_path = "/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/levenshtein_histograms/"
graph_filename = graph_path + "constitution-histogram.png"

# make histogram
df = pd.read_csv(csv_file)
df = df.sort_values('start_token')
df.to_csv(csv_file)
y = df['levenshtein_distance']
plt.figure(figsize=(20, 6))
# plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
plt.hist(y)
plt.xlabel('Levenshtein Distance')
plt.ylabel('Number of Indices')
plt.title(graph_title)
plt.savefig(graph_filename)
plt.show()