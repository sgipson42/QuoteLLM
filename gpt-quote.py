import ssl
ssl.OPENSSL_VERSION = ssl.OPENSSL_VERSION.replace("LibreSSL", "OpenSSL")
import openai
import os
import sys
import glob
import random
from Levenshtein import distance
import csv

openai.api_key = os.environ["OPENAI_API_KEY"]

completions = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a bot that produces citations for quotes."},
                  {"role": "assistant", "content": "What is the citation for this quote? Also give the surrounding context of the quote."},
                  {"role": "assistant", "content": "My fellow humans, every time I prepare for the State of the Union, I approach it with hope and expectation and excitement for our Nation. But tonight is very special, because we stand on the mountaintop of a new millennium. Behind us we can look back and see the great expanse of American achievement, and before us we can see even greater, grander frontiers of possibility. As of this date, I am hungry."}],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.0,
        )
print(completions)
sys.exit()

repetitions = 20

with open("results.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["model","file","randline","randline_count","gt","gt_portion","pred","levenshtein_distance"])

    model = "gpt-3.5-turbo"
    for transcript_file in glob.glob("transcripts/*"):
        print(transcript_file)
        with open(transcript_file) as t:
            [title, transcript] = t.read().split("\n\n", 1)
            transcript_lines = transcript.split("\n")
            for repetition in range(repetitions):
                randline = random.randint(0, len(transcript_lines))
                randline_count = random.randint(2, 6)
                gt_quote = " ".join(transcript_lines[randline:randline+randline_count]).strip()
                print(gt_quote)
                gt_words = gt_quote.split(" ")
                if len(gt_words) < 10:
                    continue
                gt_portion = random.randint(5, int(0.5*len(gt_words)))
        
                print()
                print(" ".join(gt_words[:gt_portion]))
                
                messages = [
                    {"role": "system", "content": "You are a quote generating bot. You generate quotes from well-known text sources."},
                    {"role": "assistant", "content": f"Complete this quote from {title}."},
                    {"role": "assistant", "content": " ".join(gt_words[:gt_portion])}
                    ]
        
                completions = openai.ChatCompletion.create(
                                model=model,
                                messages=messages,
                                max_tokens=1024,
                                n=1,
                                stop=None,
                                temperature=0.0,
                                )
        
                pred = completions['choices'][0]['message']['content']
                pred_words = pred.split(" ")
                trimmed_gt = gt_words[gt_portion:gt_portion+len(pred_words)]
                pred_words = pred_words[:len(trimmed_gt)] # in case gt is shorter than prediction
                print("pred:", pred_words)
                print("trimmed_gt:", trimmed_gt)
                dist = distance(" ".join(pred_words), " ".join(trimmed_gt))
                print(dist)
                print()
                csvwriter.writerow([model, title, randline, randline_count, " ".join(gt_words), " ".join(gt_words[:gt_portion]), pred, dist])

