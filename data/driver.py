import time
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver


DRIVER_PATH = "C://chromedriver/chromedriver.exe"
driver = webdriver.Chrome(DRIVER_PATH, )
driver.maximize_window()


chan_xpath = '//*[@id="channel-name"]'
subs_xpath = '//*[@id="subscriber-count"]'
videos_class = "style-scope ytd-grid-video-renderer"
views_xpath = './/*[@id="metadata-line"]/span[1]'
post_date_xpath = './/*[@id="metadata-line"]/span[2]'
title_xpath = './/*[@id="video-title"]'


def scroll_page():
    for x in range(90):
        html = driver.find_element_by_tag_name("html")
        html.send_keys(Keys.END)
        time.sleep(2)

def get_videos_from_channel(driver, channel_id):
    scroll_page()
    url = f"https://www.youtube.com/channel/{channel_id}/videos" #?view=0&sort=p&flow=grid"
    driver.get(url)
    scroll_page()
    videos = driver.find_elements_by_class_name("style-scope ytd-grid-video-renderer")
    video_lst = []
    for video in videos:
            try:
                title_element = video.find_element_by_xpath(title_xpath)
                title = title_element.text
                url = title_element.get_attribute("href")
                views = video.find_element_by_xpath(views_xpath).text
                post_date = video.find_element_by_xpath(post_date_xpath).text
                video_lst +=[{
                    "title": title,
                    "views": views,
                    "url": url,
                    "post_date": post_date
                }]
                # time.sleep(5)
            except Exception as e:
                print(e)
    return video_lst

def get_comments_from_videos(driver, video_id):
    # Source: https://github.com/dddat1017/Scraping-Youtube-Comments/blob/master/main.py

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

    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    
    try:
        username_elems = driver.find_elements_by_xpath('//*[@id="author-text"]')
        comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
    except Exception as e:
        print(e)

    comment_lst = []
    for username, comment in zip(username_elems, comment_elems):
         comment_lst += [{
             "username": username.text,
             "comment": comment.txt
         }]

    return comment_lst
    
def get_videos_from_channels(comment_lst):
    with ThreadPoolExecutor(max_workers=8) as the:
        pass


if __name__ == "__main__":
    
    #videos = get_videos_from_channel("UCeYP27qLtfUMY1b1Cyy3WdQ")
    #print(len(videos))
    comments = get_comments_from_videos("d7ShPtShl0M&t=1118s")
    print(len(comments))
    