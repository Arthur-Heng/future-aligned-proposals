import json
import os
from tqdm import tqdm
import argparse

def filter_accepted_papers(papers_path):
    paper_information = []
    # calculate the percentage of accepted papers
    accepted_papers = 0
    total_papers = 0
    percentage = 0
    with open(papers_path, "r") as f:
        for line in tqdm(f):
            line = line.strip()
            if line:
                paper = json.loads(line)
                total_papers += 1
                if 'decision' not in paper.keys() or paper['decision'] is None \
                    or 'reject' in paper['decision'].lower() or 'withdrawn' in paper['decision'].lower():
                    continue
                paper_information.append(paper)
                accepted_papers += 1
    percentage = accepted_papers / total_papers
    print(f"Total papers: {total_papers}")
    print(f"Accepted papers: {accepted_papers}")
    print(f"Percentage of accepted papers: {percentage}")
    
    basename = os.path.basename(papers_path).replace("combined_notes_", "")
    # one line one paper
    with open("data/accepted_papers/" + basename, "w") as f:
        for paper in paper_information:
            json.dump(paper, f)
            f.write("\n")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conference', type=str, default='ICLR.cc')
    parser.add_argument('--year', type=str, default='2024')
    args = parser.parse_args()

    papers_path = f"data/papers/combined_notes_{args.conference}_{args.year}.json"
    filter_accepted_papers(papers_path)