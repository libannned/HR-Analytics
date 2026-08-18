import csv
import os
import sys
from datetime import timezone

import tweepy


HANDLES = [
    "sequoia",
    "a16z",
    "ycombinator",
    "garrytan",
    "pmarca",
    "bhorowitz",
    "roelofbotha",
    "jason",
    "naval",
    "hunterwalk",
    "semil",
    "msuster",
    "pitdesi",
]

OUTPUT_FILE = "tweets_export.csv"
MAX_TWEETS_PER_HANDLE = 300
POLITICAL_KEYWORDS = [
    "trump",
    "democrat",
    "republican",
    "maga",
    "election",
    "biden",
    "political",
    "congress",
    "senate",
    "vote",
    "partisan",
]


def get_bearer_token() -> str:
    token = os.getenv("BEARER_TOKEN")
    if not token:
        raise ValueError(
            "BEARER_TOKEN environment variable is not set. "
            "Set it before running this script."
        )
    return token


def format_datetime(dt):
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def fetch_last_tweets(client: tweepy.Client, handle: str, limit: int = 100):
    user = client.get_user(username=handle)
    if user.data is None:
        print(f"No user found for @{handle}", file=sys.stderr)
        return []

    user_id = user.data.id
    collected = []

    paginator = tweepy.Paginator(
        client.get_users_tweets,
        id=user_id,
        tweet_fields=["created_at", "public_metrics", "text"],
        max_results=100,
        limit=3,
    )

    for response in paginator:
        tweets = response.data or []
        for tweet in tweets:
            tweet_text = tweet.text.replace("\n", " ").strip()
            if any(keyword in tweet_text.lower() for keyword in POLITICAL_KEYWORDS):
                continue
            metrics = tweet.public_metrics or {}
            collected.append(
                {
                    "handle": f"@{handle}",
                    "tweet text": tweet_text,
                    "date": format_datetime(tweet.created_at),
                    "likes": metrics.get("like_count", 0),
                    "retweets": metrics.get("retweet_count", 0),
                }
            )
            if len(collected) >= limit:
                return collected

    return collected


def main():
    bearer_token = get_bearer_token()

    client = tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True,
    )

    all_rows = []

    for handle in HANDLES:
        try:
            print(f"Fetching tweets for @{handle}...")
            rows = fetch_last_tweets(client, handle, MAX_TWEETS_PER_HANDLE)
            all_rows.extend(rows)
            print(f"  Retrieved {len(rows)} tweets")
        except tweepy.TweepyException as exc:
            print(f"Failed for @{handle}: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Unexpected error for @{handle}: {exc}", file=sys.stderr)

    fieldnames = ["handle", "tweet text", "date", "likes", "retweets"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Done. Wrote {len(all_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
