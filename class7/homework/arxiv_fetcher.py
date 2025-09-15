# arxiv_fetcher.py
"""Fetches recent papers from ArXiv and saves grouped summaries to a JSON file.

This script queries the ArXiv API for recent papers matching a configured search
query, groups them into batches, and writes the results to a JSON file for
subsequent processing.
"""

import json
import logging
from pathlib import Path

import arxiv
from config import Config
from logging_utils import configure_logging

log = logging.getLogger(__name__)


class ArxivFetcher:
    """Fetches and groups paper summaries from ArXiv."""

    def fetch(self):
        """Retrieve papers from ArXiv.

        Returns:
            list: A list of `arxiv.Result` objects representing the fetched papers.
        """
        log.info("Fetching papers from ArXiv...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=Config.SEARCH_QUERY,
            max_results=Config.MAX_RESULTS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        return list(client.results(search))

    def save(self, papers):
        """Save grouped paper summaries to a JSON file.

        Args:
            papers (list): List of `arxiv.Result` objects to be grouped and saved.
        """
        log.info("Saving grouped summaries...")
        groups = []
        for i in range(0, len(papers), Config.GROUP_SIZE):
            chunk = papers[i : i + Config.GROUP_SIZE]
            entries = [
                {
                    "index": i + j,
                    "title": p.title.strip(),
                    "summary": " ".join(p.summary.strip().split()),
                }
                for j, p in enumerate(chunk, start=1)
            ]
            groups.append(
                {
                    "group_index": (i // Config.GROUP_SIZE) + 1,
                    "action": Config.GROUP_PREFIX,
                    "papers": entries,
                }
            )

        output_path = Path(Config.OUTPUT_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(groups, f, indent=2, ensure_ascii=False)

        log.info(f"Saved {len(groups)} groups ({len(papers)} papers).")
        log.info(f"Output file saved at: {output_path.resolve()}")
        log.info("ArXiv fetch complete.")

    def run(self):
        """Execute the fetch and save process."""
        papers = self.fetch()
        self.save(papers)


def main():
    """Entry point for running the ArXiv fetcher as a script."""
    configure_logging()
    try:
        ArxivFetcher().run()
    except Exception as e:
        log.error(f"Unhandled error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
