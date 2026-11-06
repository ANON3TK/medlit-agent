import httpx

from medlit_agent.pubmed import PubMedArticle, PubMedClient


def test_searches_pubmed_ids() -> None:
    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith(
            "/entrez/eutils/esearch.fcgi"
        )
        assert request.url.params["db"] == "pubmed"
        assert request.url.params["term"] == "atrial fibrillation"
        assert request.url.params["retmode"] == "json"
        assert request.url.params["retmax"] == "2"

        return httpx.Response(
            200,
            json={
                "esearchresult": {
                    "idlist": [
                        "12345678",
                        "87654321",
                    ]
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
    )
    pubmed = PubMedClient(client=client)

    pubmed_ids = pubmed.search_ids(
        query="atrial fibrillation",
        limit=2,
    )

    assert pubmed_ids == [
        "12345678",
        "87654321",
    ]

def test_fetches_pubmed_article_summaries() -> None:
    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path.endswith(
            "/entrez/eutils/esummary.fcgi"
        )
        assert request.url.params["db"] == "pubmed"
        assert request.url.params["id"] == "12345678"
        assert request.url.params["retmode"] == "json"

        return httpx.Response(
            200,
            json={
                "result": {
                    "uids": ["12345678"],
                    "12345678": {
                        "uid": "12345678",
                        "title": (
                            "Atrial fibrillation "
                            "and cardiac remodeling."
                        ),
                        "pubdate": "2026 Jan",
                        "authors": [
                            {"name": "Zhang Y"},
                            {"name": "Smith J"},
                        ],
                    },
                }
            },
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handle_request),
    )
    pubmed = PubMedClient(client=client)

    articles = pubmed.fetch_summaries(
        pubmed_ids=["12345678"],
    )

    assert articles == [
        PubMedArticle(
            pubmed_id="12345678",
            title=(
                "Atrial fibrillation "
                "and cardiac remodeling."
            ),
            authors=("Zhang Y", "Smith J"),
            publication_date="2026 Jan",
            url=(
                "https://pubmed.ncbi.nlm.nih.gov/"
                "12345678/"
            ),
        )
    ]