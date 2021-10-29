"""
File containing functions to create the dataset from the two
baseline datasets described in the following papers:

Jagtap et al. (2021) https://arxiv.org/abs/2107.00941
   - we call this "baseline 1" in the code
Serrano et al. (2020) https://openreview.net/pdf?id=M4wgkxaPcyj
   - we call this "baseline 2" in the code
"""

import os
import re
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import urllib.request as request
from CaptionScraper import captionScraper
from utils import get_video_status, video2comments

# Links to raw data - baseline 1
_911_URL = "https://raw.githubusercontent.com/jagtapraj123/YT-Misinformation/main/dataset/data_911.csv"
CHEMTRAILS_URL = "https://raw.githubusercontent.com/jagtapraj123/YT-Misinformation/main/dataset/data_chemtrails.csv"
FLATEARTH_URL = "https://raw.githubusercontent.com/jagtapraj123/YT-Misinformation/main/dataset/data_flatearth.csv"
MOONLANDING_URL = "https://raw.githubusercontent.com/jagtapraj123/YT-Misinformation/main/dataset/data_moonlanding.csv"
VACCINES_URL = "https://raw.githubusercontent.com/jagtapraj123/YT-Misinformation/main/dataset/data_vaccines.csv"
DATAFILES1 = [_911_URL, CHEMTRAILS_URL, FLATEARTH_URL, MOONLANDING_URL, VACCINES_URL]

# Links to raw data - baseline 2
COMMENTS_URLS = [
    "https://raw.githubusercontent.com/JuanCarlosCSE/YouTube_misinfo/master/data/factual_comments.csv",
    "https://raw.githubusercontent.com/JuanCarlosCSE/YouTube_misinfo/master/data/labeled_comments.csv",
    "https://raw.githubusercontent.com/JuanCarlosCSE/YouTube_misinfo/master/data/non_labeled_comments.csv"
]
TITLES_URL = "https://raw.githubusercontent.com/JuanCarlosCSE/YouTube_misinfo/master/data/video_titles.csv"
VIDEOS_URL = "https://raw.githubusercontent.com/JuanCarlosCSE/YouTube_misinfo/master/data/yt_ids_conspiracy.csv"
DATAFILES2 = COMMENTS_URLS + [TITLES_URL, VIDEOS_URL]

DATAFILES = DATAFILES1 + DATAFILES2

# YouTube API developer keys
DEVELOPER_KEY_MONISH = "AIzaSyDK9A2n8Yo3tRYvHfGMEkgmilPMAE9xjMI"
DEVELOPER_KEY_ERIC = "AIzaSyDmYvkb52fq6-V5xvzYGi5jTl6KLhytyMw"

# Services
youtube_monish = build("youtube", "v3", developerKey=DEVELOPER_KEY_MONISH)
youtube_eric = build("youtube", "v3", developerKey=DEVELOPER_KEY_ERIC)


class YouTubeBaselineData(object):
    def __init__(self,
                 root_path="./tmp",
                 services=[youtube_monish, youtube_eric],
                 ) -> None:

        self.root_path = root_path
        self.services = services
        self.serv_idx = 0

        if not os.path.isdir(self.root_path):
            os.makedirs(root_path, exist_ok=True)
        
        for file in DATAFILES:
            file_csv = os.path.join(self.root_path, file.split("/")[-1])
            if not os.path.isfile(file_csv):
                request.urlretrieve(file, file_csv)

    def next_serv(self) -> None:
        """Get next service to use with YouTube Data API."""
        service = self.services[self.serv_idx]
        self.serv_idx = (self.serv_idx + 1) % len(self.services)
        return service

    #def remove_urls(x):
    #    """Remove urls from video comment."""
    #    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ", str(x))

    def get_baseline1_data(self):
        """Get working baseline 1 data, omitting removed/inaccessible videos."""

        baseline1_csv = os.path.join(self.root_path, "baseline1_videos.csv")
        
        if not os.path.isfile(baseline1_csv):
            df_overall = pd.DataFrame()

            for file in DATAFILES1:
                file_csv = os.path.join(self.root_path, file.split("/")[-1])
                df = pd.read_csv(file_csv)

                df = df.drop(columns=["Description", "vid_url"])

                # Remove removed/inaccessible videos
                upload_status = []
                for _, row in df.iterrows():
                    status = get_video_status(self.next_serv(), row["Video_ID"])
                    upload_status.append(status)
                df["upload_status"] = upload_status

                df = df[df["upload_status"] == 1]
                df = df.drop(columns=["upload_status"])

                df_overall = pd.concat([df_overall, df])

            df_overall.to_csv(baseline1_csv)
            
        return pd.read_csv(baseline1_csv)

    def get_baseline2_data(self):
        """Get working baseline 2 data, omitting removed/inaccessible videos."""

        baseline2_videos_csv = os.path.join(self.root_path, "baseline2_videos.csv")
        baseline2_comments_csv = os.path.join(self.root_path, "baseline2_comments.csv")

        if not os.path.isfile(baseline2_videos_csv):
            # Load video title/class data
            title_csv = os.path.join(self.root_path, "video_titles.csv")
            df_titles = pd.read_csv(title_csv)
            video_csv = os.path.join(self.root_path, "yt_ids_conspiracy.csv")
            df_videos = pd.read_csv(video_csv)
            df_videos = pd.merge(df_videos, df_titles, on="video_id", how="left")

            # Remove removed/inaccessible videos
            upload_status = []
            for _, row in df_videos.iterrows():
                status = get_video_status(self.next_serv(), row["video_id"])
                upload_status.append(status)
            df_videos["upload_status"] = upload_status

            df_videos = df_videos[df_videos["upload_status"] == 1]
            df_videos = df_videos.drop(columns=["upload_status"])

            # Load video comments
            factual_csv = os.path.join(self.root_path, "factual_comments.csv")
            df_comments_factual = pd.read_csv(factual_csv, sep=";")
            labeled_csv = os.path.join(self.root_path, "labeled_comments.csv")
            df_comments_labeled = pd.read_csv(labeled_csv)
            unlabeled_csv = os.path.join(self.root_path, "non_labeled_comments.csv")
            df_comments_unlabeled = pd.read_csv(unlabeled_csv)

            df_comments_factual = df_comments_factual[["video_id", "comment"]]
            df_comments_labeled = df_comments_labeled[["video_id", "comment"]]
            df_comments = pd.concat([df_comments_factual, df_comments_labeled, df_comments_unlabeled])

            #df_comments["comment"] = df_comments.comment.apply(self.remove_urls)
            
            ## Filter by first 100 comments per video
            #comments_100 = []
            #for video_id in df_videos.video_id.to_list():
            #    first_comments = df_comments[df_comments.video_id==video_id]["comment"].to_list()
            #    if len(first_comments) > 100:
            #        first_comments = first_comments[:100]
            #    comments_str = " ".join(first_comments)
            #    comments_words = re.sub(r'\W+', ' ', comments_str)
            #    comments_100.append(comments_words)
            #df_videos["comments_100"] = comments_100

            df_videos.to_csv(baseline2_videos_csv)
            df_comments.to_csv(baseline2_comments_csv)

        return pd.read_csv(baseline2_videos_csv), pd.read_csv(baseline2_comments_csv)

    def create_full_baseline1_data(self):
        """Augment baseline 1 data with baseline 2 features."""
        
        baseline1_videos_csv = os.path.join(self.root_path, "baseline1_videos.csv")
        baseline1_comments_csv = os.path.join(self.root_path, "baseline1_comments.csv")

        if not os.path.isfile(baseline1_comments_csv):

            df_videos = pd.read_csv(baseline1_videos_csv)

            comment_dict = {"video_id": [], "comment": []}
            video_ids = df_videos["Video_ID"].to_list()
            for i, video_id in enumerate(video_ids):
                print(i, "/", len(video_ids))
                try:
                    res = video2comments(self.next_serv(), video_id)
                    comment_dict["video_id"] += [video_id] * len(res[0])
                    comment_dict["comment"] += res[0]
                except HttpError:  # don't include vids with disabled comments
                    pass
                
            df_comments = pd.DataFrame(comment_dict)
            df_comments.to_csv(baseline1_comments_csv)

    def create_full_baseline2_data(self):
        """Augment baseline 2 data with baseline 1 features."""
        
        baseline2_videos_csv = os.path.join(self.root_path, "baseline2_videos.csv")
        baseline2_videos_cap_csv = os.path.join(self.root_path, "baseline2_videos_captions.csv")
        
        if not os.path.isfile(baseline2_videos_cap_csv):
            df_videos = pd.read_csv(baseline2_videos_csv)
            
            df_videos["captions"] = df_videos["video_id"].apply(captionScraper)

            # drop videos with no captions
            df_videos["captions"] = df_videos["captions"].apply(lambda x: None if x == "" else x)
            df_videos = df_videos.dropna(subset=["captions"])

            df_videos.to_csv(baseline2_videos_cap_csv)

    def create_full_data(self):
        """Return complete dataset for project experiments."""
        # Need to remove duplicate videos
        pass

if __name__ == "__main__":
    data = YouTubeBaselineData()
    data.get_baseline1_data()
    data.get_baseline2_data()
    data.create_full_baseline1_data()
    data.create_full_baseline2_data()