"""Group subquestions fetch only their own cited sources (4 Nov 2026 review fix 8).

Criteria text below is the Market Pulse 26Q4 company-revenue group's (tournament 33131), shared
by every subquestion; before the fix each company's subquestion fetched every company's page.
"""

from __future__ import annotations

from metaculus_bot.research.resolution_source import restrict_to_group_label, select_fetchable_urls

_CRITERIA = (
    "Each subquestion will resolve as the revenue reported in each company's first official quarterly "
    "earnings release after September 30, 2026. This is expected to be released no earlier than:\n\n"
    "* Apple Q4 FY2026 November 2, 2026 ([source](https://finance.yahoo.com/calendar/earnings?symbol=AAPL))\n"
    "* AMD Q3 FY2026 November 3, 2026 ([source](https://seekingalpha.com/symbol/AMD/earnings))\n"
    "* Amazon Q3 FY2026 October 29, 2026 ([source](https://seekingalpha.com/symbol/AMZN/earnings))\n"
    "* Meta Q3 FY2026 October 21, 2026 ([source](https://seekingalpha.com/symbol/META/earnings))\n"
    "* Nvidia Q3 FY 2027 November 25, 2026 ([source](https://seekingalpha.com/symbol/NVDA/earnings))\n"
    "* SpaceX Q3 FY 2026 October 20, 2026 ([source](https://www.wallstreethorizon.com/spacex-earnings-calendar))\n\n"
    "The earnings reports are typically released on the respective companies' investor relations websites "
    "and filed with the [U.S. Securities and Exchange Commission (SEC)](https://www.sec.gov/search-filings) as a Form 8-K."
)


def test_subquestion_keeps_its_own_source_and_the_shared_prose_link() -> None:
    urls = select_fetchable_urls(_CRITERIA, "", "Nvidia")
    assert "https://seekingalpha.com/symbol/NVDA/earnings" in urls
    assert "https://www.sec.gov/search-filings" in urls
    assert not any(t in u for u in urls for t in ("/AMD/", "/AMZN/", "/META/", "spacex"))


def test_label_match_is_whole_word_and_case_insensitive() -> None:
    urls = select_fetchable_urls(_CRITERIA, "", "amd")
    assert "https://seekingalpha.com/symbol/AMD/earnings" in urls
    assert "https://seekingalpha.com/symbol/AMZN/earnings" not in urls


def test_no_label_or_unmatched_label_keeps_everything() -> None:
    every = select_fetchable_urls(_CRITERIA, "")
    assert select_fetchable_urls(_CRITERIA, "", None) == every
    assert select_fetchable_urls(_CRITERIA, "", "Nov 14") == every
    assert restrict_to_group_label(_CRITERIA, "  ") == _CRITERIA
