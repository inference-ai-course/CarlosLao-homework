import arxiv
import json
import os
import requests
import trafilatura

def save_to_json(data, filename='arxiv_clean.json'):
    """
    Save paper data to a JSON file in the same directory as the script.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n Saved {len(data)} papers to {file_path}")
    except Exception as e:
        print(f"Failed to save data: {e}")

def fetch_paper_text(url, fallback_summary):
    """
    Fetch and clean full text from a URL using trafilatura.
    Falls back to the abstract if extraction fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        extracted = trafilatura.extract(response.text)
        return extracted.strip() if extracted else fallback_summary.strip()
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return fallback_summary.strip()

def main():
    """
    Fetch recent arXiv papers in cs.CL and save metadata and text.
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query="cs.CL",
        max_results=200,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []

    print("Fetching papers from arXiv...")

    for result in client.results(search):
        url = result.entry_id
        title = result.title.strip()
        summary = result.summary.strip()
        authors = [author.name for author in result.authors]
        date = result.published.strftime("%A, %B %d, %Y")

        text = fetch_paper_text(url, summary)

        paper_data = {
            "url": url,
            "title": title,
            "abstract": text,
            "authors": authors,
            "date": date
        }

        papers.append(paper_data)

    save_to_json(papers)

if __name__ == "__main__":
    main()