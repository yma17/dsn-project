from googleapiclient.discovery import build

def video2channel(service, videoid):
    """Given a video id, return the associated channel id"""
    
    request = service.videos().list(
        part="snippet",
        id=videoid
    )
    response = request.execute()

    try:
        return response["items"][0]["snippet"]["channelId"]
    except IndexError:
        return None


def channel2videos(service, channelid):
    """Given a channel id, return the associated list of video ids"""
    
    vid_ids, vid_dates, vid_titles = [], [], []
    nextPageToken = ""

    while True:
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
            break

        nextPageToken = response["nextPageToken"]
    
    return vid_ids, vid_dates, vid_titles

def video2comments(service, videoid):
    """Given a video id, return the associated top-level comments"""

    com_texts, com_dates, com_likes, com_ratings = [], [], [], []
    nextPageToken = ""

    while True:
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
            break

        nextPageToken = response["nextPageToken"]
    
    return com_texts, com_dates, com_likes, com_ratings


if __name__ == '__main__':
    DEVELOPER_KEY = 'AIzaSyDmYvkb52fq6-V5xvzYGi5jTl6KLhytyMw'
    youtube_monish = build('youtube', 'v3', developerKey=DEVELOPER_KEY)
    #print(video2channel(youtube_monish, "ngDJIjbAvz4"))
    #res, res2, res3 = channel2videos(youtube_monish, "UCoLUji8TYrgDy74_iiazvYA")
    #print(len(res), len(res2), len(res3))
    #print(res[:10])
    #print(res2[:10])
    #print(res3[:10])
    res, res2, res3, res4 = video2comments(youtube_monish, "VR8ooa3G_5M")
    print(len(res))
    print(res[:10])
    print(res2[:10])
    print(res3[:10])
    print(res4[:10])