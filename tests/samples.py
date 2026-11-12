"""Sample forwarded messages used across tests."""

import datetime as dt
from typing import Final

from scambot.factcheck_index.records import ArticleRecord

ELECTRICITY_EN: Final = (
    "Dear Consumer, your electricity power will be disconnected tonight at 9.30 PM from "
    "electricity office because your previous month bill was not updated. Please immediately "
    "contact our electricity officer 9876543210. Thank you."
)
ELECTRICITY_HI: Final = (
    "प्रिय उपभोक्ता, आपकी बिजली आज रात 9:30 बजे काट दी जाएगी क्योंकि आपका पिछले महीने का बिल "
    "अपडेट नहीं हुआ है। कृपया तुरंत बिजली अधिकारी से 9876543210 पर संपर्क करें।"
)
ELECTRICITY_HINGLISH: Final = (
    "Priya upbhokta, aapki bijli aaj raat 9:30 baje kat jayegi kyunki pichle mahine ka bill "
    "update nahi hua. Turant bijli adhikari ko 9876543210 par call karein."
)
ELECTRICITY_HINGLISH_KAAT: Final = (
    "Aapki bijli aaj raat kaat di jayegi, bill update nahi hua. Turant 9876543210 par call karein."
)
ORDINARY: Final = (
    "Happy birthday Riya! See you at the party tomorrow evening.",
    "jaldi aana, khana ready hai",
    "आज रात खाने पर आ रहे हो?",
    "Call me at 9876543210",
    "The meeting has moved to Thursday at 4 pm in the main hall.",
    "Mummy ne bola hai shaam ko sabzi le aana",
)

LAPTOP_CLAIM_HI: Final = "सरकार प्रधानमंत्री मुफ्त लैपटॉप योजना 2026 के तहत सभी छात्रों को मुफ्त लैपटॉप दे रही है"
LAPTOP_FORWARD_HI: Final = f"बड़ी खबर! {LAPTOP_CLAIM_HI}। अपने सभी दोस्तों और परिवार के ग्रुप में यह संदेश भेजें।"
LAPTOP_FACTCHECK_URL: Final = "https://factly.in/pm-free-laptop-yojana-2026-fake/"
LAPTOP_FACTCHECK: Final = ArticleRecord(
    source="factly",
    url=LAPTOP_FACTCHECK_URL,
    title="Fake news: no PM Free Laptop Yojana 2026 is giving free laptops to students",
    claim="The government is giving free laptops to all students under the PM Free Laptop "
    "Yojana 2026",
    rating="False",
    published_at=dt.datetime(2026, 10, 2, 9, 0, tzinfo=dt.UTC),
    language="en",
)
UNRELATED_FACTCHECKS: Final = (
    ArticleRecord(
        source="boom",
        url="https://www.boomlive.in/fact-check/old-flood-video-shared-as-recent",
        title="Old flood video from 2019 shared as recent",
        claim="A video shows floods in the city this week",
        rating="Misleading",
        published_at=dt.datetime(2026, 9, 12, tzinfo=dt.UTC),
        language="en",
    ),
    ArticleRecord(
        source="altnews",
        url="https://www.altnews.in/cricket-player-retirement-hoax/",
        title="Cricket captain has not announced his retirement",
        claim="The cricket captain announced his retirement from all formats",
        rating="Fake",
        published_at=dt.datetime(2026, 8, 3, tzinfo=dt.UTC),
        language="en",
    ),
)
UNRELATED_CLAIM_EN: Final = "The Taj Mahal will be closed to visitors for a year from January"
# Hindi words mapped to English concept words, so the keyword embedder links the two languages
# the way a multilingual model would.
EMBEDDING_ALIASES: Final = {
    "सरकार": ("government",),
    "प्रधानमंत्री": ("pm",),
    "मुफ्त": ("free",),
    "लैपटॉप": ("laptop", "laptops"),
    "योजना": ("scheme", "yojana"),
    "छात्रों": ("students",),
    "सभी": ("all",),
    "laptops": ("laptop",),
}
