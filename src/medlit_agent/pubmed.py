"""Client for searching biomedical literature in PubMed."""

import httpx
from dataclasses import dataclass

PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/esearch.fcgi"
)
PUBMED_SUMMARY_URL = (
    "https://eutils.ncbi.nlm.nih.gov/"
    "entrez/eutils/esummary.fcgi"
)

@dataclass(frozen=True)
class PubMedArticle:
    pubmed_id: str
    title: str
    authors: tuple[str, ...]
    publication_date: str
    url: str

class PubMedClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=30.0,
        )

    def search_ids(
        self,
        query: str,
        limit: int = 5,
    ) -> list[str]:
        if not query.strip():
            raise ValueError(
                "PubMed query must not be empty."
            )

        if limit <= 0:
            raise ValueError(
                "PubMed result limit must be positive."
            )

        response = self._client.get(
            PUBMED_SEARCH_URL,
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": limit,
                "sort": "relevance",
                "tool": "medlit_agent",
            },
        )

        response.raise_for_status()
        data = response.json()
        id_list = data["esearchresult"]["idlist"]

        return [
            str(pubmed_id)
            for pubmed_id in id_list
        ]
    def fetch_summaries(
        self,
        pubmed_ids: list[str],
    ) -> list[PubMedArticle]:
        if not pubmed_ids:
            return []

        response = self._client.get(
            PUBMED_SUMMARY_URL,
            params={
                "db": "pubmed",
                "id": ",".join(pubmed_ids),
                "retmode": "json",
                "tool": "medlit_agent",
            },
        )

        response.raise_for_status()
        data = response.json()
        result = data["result"]

        articles = []

        for pubmed_id in result["uids"]:
            summary = result[pubmed_id]

            authors = tuple(
                author["name"]
                for author in summary.get("authors", [])
                if author.get("name")
            )

            articles.append(
                PubMedArticle(
                    pubmed_id=pubmed_id,
                    title=summary.get(
                        "title",
                        "Untitled article",
                    ),
                    authors=authors,
                    publication_date=summary.get(
                        "pubdate",
                        "",
                    ),
                    url=(
                        "https://pubmed.ncbi.nlm.nih.gov/"
                        f"{pubmed_id}/"
                    ),
                )
            )

        return articles

    def search_articles(
        self,
        query: str,
        limit: int = 5,
    ) -> list[PubMedArticle]:
        pubmed_ids = self.search_ids(
            query=query,
            limit=limit,
        )

        return self.fetch_summaries(
            pubmed_ids=pubmed_ids,
        )