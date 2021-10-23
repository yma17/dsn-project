from googleapiclient.discovery import build
def video2channel(service, videoid):
    """Given a video id, return the associated channel id"""
    
    request = service.videos().list(
        part="snippet",
        id=videoid
    )
    response = request.execute()

    return response["items"][0]["snippet"]["channelId"]


def channel2videos(service, channelid, nextPageToken=""):
    """
    Given a channel id, return the associated list of video ids
    - nextPageToken is used to return the total result for a channel
    over multiple function calls
    """
    
    vid_ids, vid_dates, vid_titles = [], [], []

    request = service.search().list(
        part="snippet",
        channelId=channelid,
        order="date",
        type="video",
        maxResults=50,
        pageToken=nextPageToken
    )

    response = request.execute()
    for item in response["items"]:
        vid_ids.append(item["id"]["videoId"])
        vid_dates.append(item["snippet"]["publishedAt"])
        vid_titles.append(item["snippet"]["title"])

    if "nextPageToken" not in response.keys():
        nextPageToken = ""
    else:
        nextPageToken = response["nextPageToken"]
    
    return (vid_ids, vid_dates, vid_titles, nextPageToken)


def video2comments(service, videoid, nextPageToken=""):
    """
    Given a video id, return the associated top-level comments
    - nextPageToken is used to return the total result for a channel
    over multiple function calls
    """

    com_texts, com_dates, com_likes, com_ratings = [], [], [], []

    request = service.commentThreads().list(
        part="snippet,replies",
        videoId=videoid,
        maxResults=100,
        pageToken=nextPageToken
    )

    response = request.execute()
    for item in response["items"]:
        tlcs = item["snippet"]["topLevelComment"]["snippet"]
        com_texts.append(tlcs["textOriginal"])
        com_dates.append(tlcs["publishedAt"])
        com_likes.append(tlcs["likeCount"])
        com_ratings.append(tlcs["viewerRating"])

    if "nextPageToken" not in response.keys():
        nextPageToken = ""
    else:
        nextPageToken = response["nextPageToken"]
    
    return (com_texts, com_dates, com_likes, com_ratings, nextPageToken)


if __name__ == '__main__':
    DEVELOPER_KEY = 'AIzaSyDmYvkb52fq6-V5xvzYGi5jTl6KLhytyMw'
    youtube_monish = build('youtube', 'v3', developerKey=DEVELOPER_KEY)
    
   