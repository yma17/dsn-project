from collections import defaultdict, Counter
import json
from logging import root
from time import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import re
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from CaptionScraper import captionScraper
import urllib.request as request
import pandas as pd
import numpy as np
from utils import video2channel, channel2videos, video2comments
from driver import get_videos_from_channels, get_comments_from_videos

# Links to raw data
QUERIES_URL = (
    "https://raw.githubusercontent.com/social-comp/YouTubeAudit-data/master/queries.csv"
)
VIDEO_URL = "https://github.com/social-comp/YouTubeAudit-data/blob/master/all_results.csv?raw=true"

# YouTube API developer keys
DEVELOPER_KEY_MONISH = "AIzaSyDK9A2n8Yo3tRYvHfGMEkgmilPMAE9xjMI"
DEVELOPER_KEY_MONISH2 = "AIzaSyCYRx87mXolheLI9NP-dXS_6cHfxaFZj1g"
DEVELOPER_KEY_MONISH3 = "AIzaSyB0f_95n_gBY8cZ7ExPeo4R_aoDTMUlEwc"
DEVELOPER_KEY_MONISH4 = "AIzaSyCpaKV7mnFmn120uywoJGFhrGUw3fl1EDo"
DEVELOPER_KEY_ERIC = "AIzaSyDmYvkb52fq6-V5xvzYGi5jTl6KLhytyMw"
DEVELOPER_KEY_ASH = "AIzaSyBhsCIPer3n4C1rJT2B6n0Xt0HpX5RSvP4"

# Services
youtube_monish = build("youtube", "v3", developerKey=DEVELOPER_KEY_MONISH)
youtube_monish2 = build("youtube", "v3", developerKey=DEVELOPER_KEY_MONISH2)
youtube_monish3 = build("youtube", "v3", developerKey=DEVELOPER_KEY_MONISH3)
youtube_monish4 = build("youtube", "v3", developerKey=DEVELOPER_KEY_MONISH4)
youtube_eric = build("youtube", "v3", developerKey=DEVELOPER_KEY_ERIC)
youtube_ash = build("youtube", "v3", developerKey=DEVELOPER_KEY_ASH)

tqdm.pandas()


class YoutubeAuditData(object):
    def __init__(
        self,
        root_path="./tmp",
        services=[youtube_monish,
                  youtube_monish2,
                  youtube_monish3,
                  youtube_monish4,
                  youtube_eric,
                  youtube_ash],
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
        self.root_path = root_path
        self.service_idx = 0

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
        vids_caps_csv = os.path.join(root_path, "vids_with_caps.csv")
        if not os.path.isfile(caption_csv) or not os.path.isfile(vids_caps_csv):
            caption_df = video_df[["video_id"]].copy()

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as the:
                res = the.map(captionScraper, caption_df["video_id"].to_list())
            caption_df["captions"] = pd.Series((v for v in res))

            # drop videos with no captions
            caption_df["captions"] = caption_df["captions"].apply(lambda x: None if x == "" else x)
            caption_df = caption_df.dropna(subset=["captions"])
            
            caption_df.to_csv(caption_csv, index=False)

            vids_with_caps = caption_df[["video_id"]]
            vids_with_caps.to_csv(vids_caps_csv, index=False)
        else:
            vids_with_caps = pd.read_csv(vids_caps_csv)

        vids_with_caps = vids_with_caps["video_id"].to_list()  # ids of vids with captions
        caption_df = None  # clear memory

        num_comment_files = len(glob(os.path.join(root_path, "comment_audit_*.csv")))
        comment_csv = os.path.join(root_path, "comment_audit_{}.csv".format(num_comment_files))
        print(comment_csv)
        comment_state = os.path.join(root_path, "comment_state.json")
        if not os.path.isfile(comment_state):
            self.vid_idx = 0
            self.comment_next_page_token = None
        else:
            with open(comment_state, "r") as f:
                state = json.load(f)
                self.vid_idx = state["vid_idx"]
                self.comment_next_page_token = state["comment_next_page_token"]
        comment_data = {"text": [], "datetime": [], "likes": [],
                        "viewer_rating": [], "video_id": []}

        # Iterate through videos (with captions) and get comments
        print("Starting at index", self.vid_idx)
        while self.vid_idx < len(vids_with_caps):
            if self.vid_idx % 100 == 0:
                print(self.vid_idx)

            video_id = vids_with_caps[self.vid_idx]
            res_comment = self.get_comments_from_video(video_id)

            if not res_comment:  # if encounter an error with all api keys
                # quota limit (probably) exceeded, save state and save csv.
                with open(comment_state, "w") as f:
                    json.dump({"vid_idx": self.vid_idx,
                        "comment_next_page_token":self.comment_next_page_token}, f)
                    comment_df = pd.DataFrame(comment_data)
                    comment_df = comment_df.dropna()
                    comment_df.to_csv(comment_csv, index=False, sep='\t')
                    print("Ended at index", self.vid_idx)
                    break

            # Remove \r from comments
            for i in range(len(res_comment[0])):
                res_comment[0][i] = res_comment[0][i].replace("\r", " ")

            comment_data["text"] += res_comment[0]
            comment_data["datetime"] += res_comment[1]
            comment_data["likes"] += res_comment[2]
            comment_data["viewer_rating"] += res_comment[3]
            comment_data["video_id"] += [video_id] * len(res_comment[0])

            if len(res_comment[0]) > 0:
                self.comment_next_page_token = res_comment[4]
            
            self.vid_idx += 1
        
        comment_all_unclean = os.path.join(root_path, "comment_auditunclean.csv")
        if self.vid_idx == len(vids_with_caps) and not os.path.isfile(comment_all_unclean):
            print("Finished collecting all comments")
            with open(comment_state, "w") as f:
                json.dump({"vid_idx": self.vid_idx,
                    "comment_next_page_token":self.comment_next_page_token}, f)
                comment_df = pd.DataFrame(comment_data)
                comment_df = comment_df.dropna()
                comment_df.to_csv(comment_csv, index=False, sep='\t')
            dfs = []
            for filename in glob(os.path.join(root_path, "comment_audit_*.csv")):
                df = pd.read_csv(filename, sep='\t')
                dfs += [df]
            df_all = pd.concat(dfs, axis=0)
            df_all.to_csv(comment_all_unclean, index=False, sep='\t')
        
        # Clean all comments retrieved
        comment_all = os.path.join(root_path, "comment_audit.csv")
        if not os.path.isfile(comment_all):
            df_all = None
            df_all = pd.read_csv(comment_all_unclean, sep='\t')

            df_all = df_all.dropna(axis=0, how='any')

            def remove_urls(x):
                return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " ", str(x))
            
            df_all['text'] = df_all['text'].apply(remove_urls)

            def remove_whitespace_comments(row):
                return not row['text'].isspace()

            m = df_all.apply(remove_whitespace_comments, axis=1)
            df_all = df_all[m]

            df_all.to_csv(comment_all, index=False, sep='\t')
        else:
            df_all = pd.read_csv(comment_all, sep='\t')

        # Print statistics of final scrapped data
        print("Printing statistics:")
        print("Columns:", df_all.columns)
        print("Counting null values:", df_all.isna().sum())
        print("Total comments:", len(df_all))
        print("Earliest comment datetime:", df_all['datetime'].min())
        print("Latest comment datetime:", df_all['datetime'].max())
        print("Comment distribution by year:")
        z = [s[:4] for s in df_all['datetime'].to_list()]
        print(Counter(z))
        print("Comment distribution by likes:")
        z = np.array([int(i) for i in df_all['likes'].to_list()])
        print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())
        print("Comment distribution by viewer rating:")
        z = [b for b in df_all['viewer_rating'].to_list()]
        print(Counter(z))
        print("Comment distribution by number of tokens:")
        z = np.array([len(s.split(' ')) for s in df_all['text']])
        print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())
        print("Number of unique videos:", len(set(df_all['video_id'].to_list())))
        print("Num comments distribution by video:")
        z = df_all.groupby(['video_id']).size().to_numpy()
        print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())
        video_df = pd.read_csv(video_id_csv)
        video_df_final = video_df[video_df["video_id"].isin(df_all["video_id"])]
        channels_final = set(video_df_final["channel_id"].to_list())
        print("Number of unique channels:", len(channels_final))
        print("Num videos distribution by channel:")
        z = video_df_final.groupby(['channel_id']).size().to_numpy()
        print("min,max,avg,std", z.min(), z.max(), z.mean(), z.std())


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
    
    def get_comments_from_video(self, video_id, mult_calls=False) -> tuple:
        """Get a set of comment data given a video id. Return results."""

        res = None
        for _ in range(len(self.services)):  # cycle through api keys until successful
            try:
                if mult_calls:
                    res = video2comments(self.services[self.service_idx], video_id, self.comment_next_page_token)
                else:
                    res = video2comments(self.services[self.service_idx], video_id)
                self.update_service_idx()
                break
            except Exception as e:
                message = str(e)
                #print(message)
                self.update_service_idx()
                if "\'reason\': \'commentsDisabled\'" in message:
                    print("Note: comments disabled for video", video_id)
                    res = ([], [], [], [], None)
                    break
                elif "\'reason\': \'videoNotFound\'" in message:
                    print("Note: video", video_id, "could not be found")
                    res = ([], [], [], [], None)
                    break
                elif "\'reason\': \'channelNotFound\'" in message:
                    print("Note: channel corresponding to video", video_id, "could not be found")
                    res = ([], [], [], [], None)
                    break
                elif "\'reason\': \'quotaExceeded\'" in message:
                    pass
                else:
                    pass
        
        return res


if __name__ == "__main__":
    data = YoutubeAuditData()
