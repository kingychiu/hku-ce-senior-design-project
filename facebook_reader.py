import requests
from file_io import FileIO
base_path = "https://graph.facebook.com"
token = "1825924290976649|f4a421b77888587f351418a5aa84762c"
# page_ids = ['BillGates', 'elonmuskofficial', 'emmawatson', 'barackobama', 'DonaldTrump']
# page_ids = ['emmawatson', 'barackobama', 'DonaldTrump']
page_ids = ['JKRowling', 'oprahwinfrey']


for page_id in page_ids:
    texts = []
    feedRequestUrl = base_path + "/" + page_id + "/feed?access_token=" + token

    def do(url):
        r = requests.get(url)
        json_dict = r.json()
        for data in json_dict['data']:
            print(data.keys())
            if 'message' in data.keys():
                data = data['message'].replace('\n', ' ')
                if len(data) >= 50:
                    texts.append(data)
        print(len(texts))
        if len(texts) >= 1000:
            return
        if 'paging' in json_dict.keys():
            do(url)
    do(feedRequestUrl)
    # write text
    FileIO.write_lines_to_file('./fb_posts/'+page_id+'.txt', texts)
