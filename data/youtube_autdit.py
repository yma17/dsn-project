from collections import defaultdict
from logging import root
from googleapiclient.discovery import build
import os
import urllib.request as request
import pandas as pd
from utils import video2channel, channel2videos 

queries_url = 'https://raw.githubusercontent.com/social-comp/YouTubeAudit-data/master/queries.csv'
video_url = "https://github.com/social-comp/YouTubeAudit-data/blob/master/all_results.csv?raw=true"



DEVELOPER_KEY_MONISH = 'AIzaSyDK9A2n8Yo3tRYvHfGMEkgmilPMAE9xjMI'
DEVELOPER_KEY_ERIC = 'AIzaSyDmYvkb52fq6-V5xvzYGi5jTl6KLhytyMw'

youtube_monish = build('youtube', 'v3', developerKey=DEVELOPER_KEY_MONISH)
youtube_eric = build('youtube', 'v3', developerKey=DEVELOPER_KEY_ERIC)
# categories_request = youtube_monish.videoCategories().list(part='snippet', regionCode='US', hl='en_US')
#channel_request = youtube_monish.channels().list(part=['id', 'topicDetails', 'snippet'], )

class YoutubeAuditData(object):
    def __init__(self, root_path='./tmp', services=[youtube_monish]) -> None:
        service_idx = 0
        if not os.path.isdir(root_path):
            os.makedirs(root_path, exist_ok=True)
        vid_ids_csv = os.path.join(root_path, 'vid_url.csv')
        if not os.path.isfile(vid_ids_csv):
            queries_csv = os.path.join(root_path, 'queries.csv')
            if not os.path.isfile(queries_csv):
                request.urlretrieve(queries_url, queries_csv)

            video_url_csv = os.path.join(root_path, 'all_results.csv')
            if not os.path.isfile(video_url_csv):
                request.urlretrieve(video_url, video_url_csv)
            
            video_url_csv = pd.read_csv(video_url_csv)
            vid_ids_df = video_url_csv['vid_url'].apply(lambda x: x.split('=')[-1])
            vid_ids_df.to_csv(vid_ids_csv)

        vid_ids_df = pd.read_csv(vid_ids_csv)['vid_url']
        vid_ids_df = set(vid_ids_df.to_list())
        channel_sampling = defaultdict(list)
        video_error_count = 0

        channel_id_csv = os.path.join(root_path, 'channel_id.csv')
        


        if not os.path.isfile(channel_id_csv):
            for vid_url in vid_ids_df:
                channel_id = video2channel(services[service_idx], vid_url)
                if channel_id is not None:
                    channel_sampling[channel_id] += [vid_url]
                else:
                    video_error_count += 1
   
            channel_sampling_dump = {'channel_ids':[], 'vid_urls':[]}
            for k, v in channel_sampling.items():
                channel_sampling_dump['channel_ids'].append(k)
                channel_sampling_dump['vid_urls'].append(v)

            channel_id_df = pd.DataFrame(channel_sampling_dump)
            channel_id_df.to_csv(channel_id_csv)
        channel_id_df = pd.read_csv(channel_id_csv)

        video_channel_csv = os.path.join(root_path, 'video_channel_ids.csv')
        if not os.path.isfile(video_channel_csv):
            video_channel_dict = {'video_id':[], 'video_date':[], 'video_title': [], 'channel_id':[]}

            for channel_id in channel_id_df['channel_ids']:
                try:
                    video = channel2videos(services[service_idx], channel_id)                
                except:
                    video = channel2videos(service[service_idx], channel_id)
                video_channel_dict['video_id'] += video[0]
                video_channel_dict['video_date'] += video[1]
                video_channel_dict['video_title'] += video[2]
                video_channel_dict['channel_id'] += [channel_id] * len(video[0]) 
            
            video_channel_df = pd.DataFrame(video_channel_dict)
            video_channel_df.to_csv(video_channel_csv)
        video_channel_df = pd.read_csv(video_channel_csv)
        

if __name__ == "__main__":
    data = YoutubeAuditData(service=[youtube_monish,youtube_eric])
    print()

    