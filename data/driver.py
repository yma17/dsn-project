import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

DRIVER_PATH = "C://chromedriver/chromedriver.exe"


chan_xpath = '//*[@id="channel-name"]'
subs_xpath = '//*[@id="subscriber-count"]'
videos_class = "style-scope ytd-grid-video-renderer"
views_xpath = './/*[@id="metadata-line"]/span[1]'
post_date_xpath = './/*[@id="metadata-line"]/span[2]'
title_xpath = './/*[@id="video-title"]'
WINDOW_SIZE = "1920,1080"


def scroll_page(driver):
    for x in range(30):
        html = driver.find_element_by_tag_name("html")
        html.send_keys(Keys.END)
        time.sleep(2)

def get_videos_from_channel(channel_id):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    driver = webdriver.Chrome(DRIVER_PATH, options=chrome_options)

    scroll_page(driver)
    url = f"https://www.youtube.com/channel/{channel_id}/videos" #?view=0&sort=p&flow=grid"
    driver.get(url)
    scroll_page(driver)
    videos = driver.find_elements_by_class_name("style-scope ytd-grid-video-renderer")
    video_lst = []
    for video in videos:
            try:
                title_element = video.find_element_by_xpath(title_xpath)
                title = title_element.text
                url = title_element.get_attribute("href")
                url_parse = urlparse(url)
                query = parse_qs(url_parse.query)
                video_id = query["v"][0]

                views = video.find_element_by_xpath(views_xpath).text
                post_date = video.find_element_by_xpath(post_date_xpath).text
                video_lst +=[{
                    "channel_id": channel_id,
                    "title": title,
                    "views": views,
                    "url": url,
                    "video_id": video_id,
                    "post_date": post_date
                }]
                # time.sleep(5)
            except Exception as e:
                print(e)
    driver.close()
    return video_lst

def get_comments_from_videos(video_id):
    # Source: https://github.com/dddat1017/Scraping-Youtube-Comments/blob/master/main.py
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    driver = webdriver.Chrome(DRIVER_PATH, options=chrome_options)
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    driver.get(url)  # navigate to url
    time.sleep(5)  # give time for page to load

    try:  # extract comments
        comment_section = driver.find_element_by_xpath('//*[@id="comments"]')
    except Exception as e:
        print(e)

    # Scroll into view the comment section, then allow some time
    # for everything to be loaded as necessary.
    # scroll_page()
    driver.execute_script("arguments[0].scrollIntoView();", comment_section)
    time.sleep(7)

    # last_height = driver.execute_script("return document.documentElement.scrollHeight")

    # while True:
    #     driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    #     time.sleep(2)
    #     new_height = driver.execute_script("return document.documentElement.scrollHeight")
    #     if new_height == last_height:
    #         break
    #     last_height = new_height

    # driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    
    scroll_page() 

    try:
        username_elems = driver.find_elements_by_xpath('//*[@id="author-text"]')
        comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
    except Exception as e:
        print(e)

    comment_lst = []
    for username, comment in zip(username_elems, comment_elems):
         comment_lst += [{
             "video_id": video_id,
             "username": username.text,
             "comment": comment.text,
         }]

    driver.close()
    return comment_lst
    
def get_videos_from_channels(channel_lst):
    video_lst = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as the:
        video_lst = the.map(get_videos_from_channel, channel_lst)
    return video_lst

def get_comments_from_videos(video_lst):
    comment_lst = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as the:
        comment_lst = the.map(get_comments_from_videos, video_lst)
    return comment_lst

test_set = ['UC-SJ6nODDmufqBzPBwCvYvQ',
'UCoLrcjPV5PbUrUyXq5mjc_A',
'UCQfKpN66UN8d5Z15gFV9PGQ',
'UC7_gcs09iThXybpVgjHZ_7g',
'UCyaSMsSwOYgpB_68feSuezQ',
'UCtu2THvit1LjLONLqvD8uAQ',
'UCS_H_4AmsqC705DObesZIIg',
'UCeiYXex_fwgYDonaTcSIk6w',
'UC8p1vwvWtl6T73JiExfWs1g',
'UCX7KkEUl8gP3ww7uq-wS7AA',
'UCIHdDJ0tjn_3j-FS7s_X1kQ',
'UCcjoLhqu3nyOFmdqF17LeBQ',
'UCRtsZ5Iak9wSLsQLQ3XOAeA',
'UC1h3juWLuF-Z-KSJd8GErtw']

if __name__ == "__main__":
    video_dict = get_videos_from_channels(test_set)
    # driver = webdriver.Chrome(DRIVER_PATH, )
    # driver.maximize_window()

    # #videos = get_videos_from_channel(driver, "UCeYP27qLtfUMY1b1Cyy3WdQ")
    # #print(len(videos))
    # comments = get_comments_from_videos(driver, "d7ShPtShl0M&t=1118s")
    # print(len(comments))

    


    print()
    