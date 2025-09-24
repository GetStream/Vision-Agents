from getstream import Stream
from dotenv import load_dotenv
from getstream.models import QueryReviewQueueResponse, ReviewQueueItemResponse

load_dotenv()


if __name__ == "__main__":
    client = Stream.from_env()

    response: QueryReviewQueueResponse = client.moderation.query_review_queue().data
    item: ReviewQueueItemResponse
    for item in response.items:
        print("--------------------------------")
        print(f"Flagged audio transcript with ID: {item.id}")
        print(f"Recommended action: {item.recommended_action}")
        if item.recommended_action != "keep":
            print(f"Transcript: {item.moderation_payload.texts[0]}")
        print(f"Created at: {item.created_at.isoformat()}")

        # Mark the item as reviewed
        client.moderation.submit_action("mark_reviewed", item.id, user_id=item)
