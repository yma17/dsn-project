from selenium import webdriver
import pandas as pd

DRIVER_PATH = "/usr/local/bin/chromedriver"

CHANNEL_ID_CSV = "./tmp/channel_id.csv"
CHANNEL_VERDICTS_CSV = "./tmp/channel_verdicts.csv"

# "Will this channel contain info relevant to our topics?"
# 1 - most likely not
# 2 - probably not
# 3 - netural/unsure
# 4 - probably yes
# 5 - most likely yes
VERDICTS = ["1", "2", "3", "4", "5"]

def get_channel_verdict(channel_id):
    print("CURRENT CHANNEL BEING LOOKED AT", channel_id)
    driver = webdriver.Chrome(DRIVER_PATH)

    url = f"https://www.youtube.com/channel/{channel_id}/videos"
    driver.get(url)

    while True:
        v = input()
        if v in VERDICTS:
            break
        else:
            print("Invalid input, enter again")

    driver.close()
    return v

def get_channel_verdicts(channel_lst):
    verdict_lst = []
    for i, channel_id in enumerate(channel_lst):
        if i % 20 == 0:
            # Print summary of verdicts so far
            stats_str = "Summary at channel {}/{}: ".format(i, len(channel_lst))
            for v in VERDICTS:
                stats_str += "{}: {}, ".format(v, verdict_lst.count(v))
            print(stats_str[:-2])
            # Save verdicts so far to file (in case any)
            df = pd.DataFrame({"channel_id": channel_lst[:i], "verdict": verdict_lst})
            df.to_csv(CHANNEL_VERDICTS_CSV)
        verdict_lst += [get_channel_verdict(channel_id)]
    return verdict_lst

if __name__ == '__main__':
    channel_lst = list(pd.read_csv(CHANNEL_ID_CSV)["channel_ids"])
    #channel_lst = channel_lst[:5]
    verdict_lst = get_channel_verdicts(channel_lst)
    df = pd.DataFrame({"channel_id": channel_lst, "verdict": verdict_lst})
    df.to_csv(CHANNEL_VERDICTS_CSV)