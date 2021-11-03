from collections import defaultdict
import json
from logging import root
from time import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from CaptionScraper import captionScraper
import urllib.request as request
import pandas as pd
from utils import video2channel, channel2videos, video2comments
from driver import get_videos_from_channels, get_comments_from_videos

# Links to raw data
QUERIES_URL = (
    "https://raw.githubusercontent.com/social-comp/YouTubeAudit-data/master/queries.csv"
)
VIDEO_URL = "https://github.com/social-comp/YouTubeAudit-data/blob/master/all_results.csv?raw=true"

# YouTube API developer keys

# Services
youtube_monish = build("youtube", "v3", developerKey=DEVELOPER_KEY_MONISH)
youtube_eric = build("youtube", "v3", developerKey=DEVELOPER_KEY_ERIC)
youtube_ash = build("youtube", "v3", developerKey=DEVELOPER_KEY_ASH)

tqdm.pandas()


class YoutubeAuditData(object):
    def __init__(
        self,
        load_state=False,
        root_path="./tmp",
        services=[youtube_monish, youtube_eric, youtube_ash],
    ) -> None:
    
        if not os.path.isdir(root_path):
            os.makedirs(root_path, exist_ok=True)

        queries_csv = os.path.join(root_path, "queries.csv")
        if not os.path.isfile(queries_csv):
            request.urlretrieve(QUERIES_URL, queries_csv)

        video_url_csv = os.path.join(root_path, "all_results.csv")
        if not os.path.isfile(video_url_csv):
            request.urlretrieve(VIDEO_URL, video_url_csv)

        self.services = services

        if load_state and os.path.isfile("./state_json"):
            self.load_state()
        else:
            self.root_path = root_path
            self.service_idx = 0

            # for channel2videos
            self.curr_channel_idx = 0
            self.curr_video_data = None
            self.curr_video_idx = 0
            self.video_next_page_token = ""

            # for videos2comments
            self.curr_comment_data = None
            self.comment_next_page_token = ""

        vid_ids_csv = self.get_vid_ids(video_url_csv)
        channel_id_csv = self.get_channel_ids(vid_ids_csv)
        channel_lst = list(pd.read_csv(channel_id_csv)["channel_ids"])
        
        video_id_csv = os.path.join(root_path, "video_id.csv")
        if not os.path.isfile(video_id_csv):                
            channel_video_lst = get_videos_from_channels(channel_lst) # channel x videos x dict[5]
            video_tuple = {"channel_id":[], "video_id":[], "title":[], "views": [], "url":[], "post_date":[]}
            for i, video_lst in enumerate(channel_video_lst):
                for video in video_lst:
                    video_tuple['channel_id'] += [video['channel_id']]
                    video_tuple['video_id'] += [video['video_id']]
                    video_tuple['title'] += [video['title']]
                    video_tuple['views'] += [video['views']]
                    video_tuple['url'] += [video['url']]                
                    video_tuple['post_date'] += [video['post_date']]
                
            video_df = pd.DataFrame(video_tuple)
            video_df.to_csv(video_id_csv)
        else:
            video_df = pd.read_csv(video_id_csv)

        caption_csv = os.path.join(root_path, "caption.csv")
        if not os.path.isfile(caption_csv):
            caption_df = video_df[["video_id"]].copy()

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as the:
                res = the.map(captionScraper, caption_df["video_id"].to_list())
            caption_df["captions"] = pd.Series((v for v in res))

            # drop videos with no captions
            caption_df["captions"] = caption_df["captions"].apply(lambda x: None if x == "" else x)
            caption_df = caption_df.dropna(subset=["captions"])
            
            caption_df.to_csv(caption_csv, index=False)
        else:
            caption_df = pd.read_csv(caption_csv)

        #comment_csv = os.path.join(root_path, "comment.csv")
        #if not os.path.isfile(comment_csv):
        #    comment_tuple = {"video_id":[], "username":[], "comment":[]}
        #    for i, videos in enumerate(video_lst):
        #        comment_lst = get_comments_from_videos(videos)
        #        comment_tuple['video_id'] += [c['video_id'] for c in comment_lst]
        #        comment_tuple['username'] += [c['username'] for c in comment_lst]
        #        comment_tuple['comment'] += [c['comment'] for c in comment_lst]
        #    comment_df = pd.DataFrame(comment_tuple)
        #    comment_df.to_csv(comment_csv)
        #else:
        #    comment_df = pd.read_csv(comment_csv)


    def save_state(self, path="./state.json") -> None:
        """Save data creation state to .json file."""

        state_dict = {
            "root_path": self.root_path,
            "service_idx": self.service_idx,
            "curr_channel_idx": self.curr_channel_idx,
            "curr_video_data": self.curr_video_data,
            "curr_video_idx": self.curr_video_idx,
            "video_next_page_token": self.video_next_page_token,
            "curr_commment_data": self.curr_comment_data,
            "comment_next_page_token": self.comment_next_page_token,
        }
        with open(path, "w") as f:
            json.dump(state_dict, f)

    def load_state(self, path="./state.json") -> None:
        """Load data creation state from .json file."""
        with open(path, "r") as f:
            state_dict = json.load(f)
            self.root_path = state_dict["root_path"]
            self.service_idx = state_dict["service_idx"]
            self.curr_channel_idx = state_dict["curr_channel_idx"]
            self.curr_video_data = state_dict["curr_video_data"]
            self.curr_video_idx = state_dict["curr_video_idx"]
            self.video_next_page_token = (state_dict["video_next_page_token"],)
            self.curr_comment_data = (state_dict["curr_comment_data"],)
            self.comment_next_page_token = state_dict["comment_next_page_token"]

    def update_service_idx(self) -> None:
        """Increment service index."""
        self.service_idx = (self.service_idx + 1) % len(self.services)

    def get_vid_ids(self, video_url_csv) -> str:
        """
        Retrieve video ids from original dataset; store in .csv file.
        Return path to stored file.
        """

        vid_ids_csv = os.path.join(self.root_path, "vid_url.csv")
        if not os.path.isfile(vid_ids_csv):
            video_url_csv = pd.read_csv(video_url_csv)
            vid_ids_df = video_url_csv["vid_url"].apply(lambda x: x.split("=")[-1])

            vid_ids_df.to_csv(vid_ids_csv)
        return vid_ids_csv

    def get_channel_ids(self, vid_ids_csv, print_interval=100) -> str:
        """
        Retrieve channel ids from videos in original dataset; store in .csv file.
        Return path to stored file.
        """

        vid_ids_df = pd.read_csv(vid_ids_csv)["vid_url"]
        vid_ids_df = list(set(vid_ids_df.to_list()))  # list of unique vid ids
        channel_sampling = defaultdict(list)

        channel_id_csv = os.path.join(self.root_path, "channel_id.csv")
        if not os.path.isfile(channel_id_csv):
            video_error_count = 0
            idx = 0
            while idx < len(vid_ids_df):
                if idx % print_interval == 0:
                    print("get_channel_ids:", idx, "/", len(vid_ids_df))
                vid_url = vid_ids_df[idx]
                try:
                    channel_id = video2channel(self.services[self.service_idx], vid_url)
                    channel_sampling[channel_id] += [vid_url]
                except IndexError:  # video does not exist
                    video_error_count += 1
                except HttpError:
                    idx -= 1  # redo

                self.update_service_idx()
                idx += 1

            channel_sampling_dump = {"channel_ids": [], "vid_urls": []}
            for k, v in channel_sampling.items():
                channel_sampling_dump["channel_ids"].append(k)
                channel_sampling_dump["vid_urls"].append(v)

            channel_id_df = pd.DataFrame(channel_sampling_dump)
            channel_id_df.to_csv(channel_id_csv)

            print("Video error count =", video_error_count)

        return channel_id_csv

    def get_videos_and_comments(self, channel_id_csv, get_videos=True) -> None:
        """TODO: add docstring"""

        video_channel_csv = os.path.join(self.root_path, "video_channel.csv")
        comment_video_csv = os.path.join(self.root_path, "comment_video.csv")

        # Load from existing csv files if they exist
        if os.path.isfile(video_channel_csv):
            video_channel_data = pd.read_csv(video_channel_csv)
            video_channel_data = video_channel_data.reset_index().to_dict(orient="list")
        else:
            video_channel_data = {
                "video_id": [],
                "video_date": [],
                "video_title": [],
                "channel_id": [],
            }
        if os.path.isfile(comment_video_csv):
            comment_video_data = pd.read_csv(comment_video_csv)
            comment_video_data = comment_video_data.reset_index().to_dict(orient="list")
        else:
            comment_video_data = {
                "comment_text": [],
                "comment_datetime": [],
                "comment_likes": [],
                "comment_viewer_rating": [],
                "video_id": [],
            }

        def write_to_file():
            video_channel_df = pd.DataFrame(video_channel_data)
            video_channel_df.to_csv(video_channel_csv, index=False, sep="\t")
            comment_video_df = pd.DataFrame(comment_video_data)
            comment_video_df.to_csv(comment_video_csv, index=False)
            self.save_state()

        channel_df = pd.read_csv(channel_id_csv)
        channel_ids = channel_df["channel_ids"].to_list()


        while self.curr_channel_idx < len(channel_ids):  # for each channel
            channel_id = channel_ids[self.curr_channel_idx]
            while True:  # get all vids across iterations
                if not self.curr_video_data:
                    res_video = self.get_videos_from_channel(channel_id)  # get a set of vids using API
                    if not res_video:  # if encounter an error with all api keys
                        print("Operation failed with all api keys - they may be broken or quota limit exceeded.")
                        write_to_file()
                        return
                else:  # recover data obtained from previous run
                    res_video = self.curr_video_data
                self.curr_video_data = res_video
                self.curr_video_idx = 0
                self.video_next_page_token = res_video[3]

                video_channel_data["video_id"] += res_video[0]
                video_channel_data["video_date"] += res_video[1]
                video_channel_data["video_title"] += res_video[2]
                video_channel_data["channel_id"] += [channel_id] * len(res_video[0])

                while self.curr_video_idx < len(self.curr_video_data):  # for each video
                    video_id = res_video[0][self.curr_video_idx]
                    while True:  # get all comments across iterations
                        if not self.curr_comment_data:
                            res_comment = self.get_comments_from_video(video_id)  # get a set of comments using API

                            if not res_comment:  # if encounter an error with all api keys
                                print("Operation failed with all api keys - they may be broken or quota limit exceeded.")
                                write_to_file()
                                return
                        else:  # recover data obtained from previous run
                            res_comment = self.curr_comment_data

                        self.curr_comment_data = res_comment
                        self.comment_next_page_token = res_comment[4]

                        # Update content .csv contents
                        comment_video_data["comment_text"] += res_comment[0]
                        comment_video_data["comment_datetime"] += res_comment[1]
                        comment_video_data["comment_likes"] += res_comment[2]
                        comment_video_data["comment_viewer_rating"] += res_comment[3]
                        comment_video_data["video_id"] += [video_id] * len(res_comment[0])

                        if self.comment_next_page_token == "":  # no more comments to get
                            self.curr_comment_data = None
                            break

                    self.curr_video_idx += 1  # next video

                if self.video_next_page_token == "":  # no more vids to get
                    self.curr_video_data = None
                    break

            self.curr_channel_idx += 1  # next channel

        write_to_file()

    def get_videos_from_channel(self, channel_id) -> tuple:
        """Get a set of video data given a channel id. Return results."""

        res = None
        for _ in range(len(self.services)):  # cycle through api keys until successful
            try:
                res = channel2videos(self.services[self.service_idx], channel_id, self.video_next_page_token)
                break
            except:
                pass
            self.update_service_idx()

        return res

    def get_comments_from_video(self, video_id) -> tuple:
        """Get a set of comment data given a video id. Return results."""

        res = None
        for _ in range(len(self.services)):  # cycle through api keys until successful
            try:
                res = video2comments(self.services[self.service_idx], video_id, self.comment_next_page_token)
                break
            except:
                pass
            self.update_service_idx()

        return res


if __name__ == "__main__":
    # data = YoutubeAuditData(services=[youtube_monish,youtube_eric,youtube_ash])
    # data.save_state()
    data = YoutubeAuditData(load_state=True)
