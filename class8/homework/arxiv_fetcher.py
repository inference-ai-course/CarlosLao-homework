# arxiv_fetcher.py
"""Fetches recent papers from ArXiv and downloads PDFs to a raw folder.

This script queries the ArXiv API for recent papers matching a configured search
query, downloads their PDFs, and stores them under the configured raw folder.
"""

import logging
import shutil
from pathlib import Path

import arxiv
from config import Config
from logging_utils import configure_logging

log = logging.getLogger(__name__)


class ArxivFetcher:
    """Fetches papers from ArXiv and downloads their PDFs."""

    def clear_output_folder(self) -> None:
        """Ensure the output folder exists and is empty.

        Removes existing files and directories under the raw folder to provide a clean
        workspace for new downloads. Creates the folder if it does not exist.
        """
        if Config.RAW_FOLDER.exists():
            for item in Config.RAW_FOLDER.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    log.warning("Failed to remove %s: %s", item, e)
        else:
            Config.RAW_FOLDER.mkdir(parents=True, exist_ok=True)
        log.info("Output folder ready: %s", Config.RAW_FOLDER)

    def download_pdf(self, result: arxiv.Result, path: Path) -> bool:
        """Download a single paper's PDF to the specified path.

        Args:
            result (arxiv.Result): The arXiv result representing a paper.
            path (Path): Destination path for the downloaded PDF.

        Returns:
            bool: True if the PDF is present or successfully downloaded; False otherwise.
        """
        if path.exists():
            return True
        try:
            result.download_pdf(filename=path)
            return True
        except Exception as e:
            log.warning("Download failed %s: %s", result.get_short_id(), e)
            return False

    def fetch(self) -> list[Path]:
        """Query arXiv and download PDFs for matched results.

        Returns:
            list[Path]: List of paths to downloaded PDFs within the raw folder.
        """
        search = arxiv.Search(
            query=Config.ARXIV_QUERY,
            max_results=Config.MAX_RESULTS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        client = arxiv.Client()
        pdfs: list[Path] = []
        for res in client.results(search):
            path = Config.RAW_FOLDER / f"{res.get_short_id()}.pdf"
            if self.download_pdf(res, path):
                pdfs.append(path)
        return pdfs

    def run(self) -> list[Path]:
        """Execute the clear, fetch, and download process.

        Returns:
            list[Path]: List of paths to the downloaded PDFs.
        """
        self.clear_output_folder()
        log.info("Fetching arXiv papers...")
        pdfs = self.fetch()
        log.info("Stored %d PDFs in %s", len(pdfs), Config.RAW_FOLDER.resolve())
        return pdfs


def main() -> None:
    """Entry point for running the ArXiv fetcher as a script."""
    configure_logging()
    try:
        ArxivFetcher().run()
    except Exception as e:
        log.error("Unhandled error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
