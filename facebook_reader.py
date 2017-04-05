import requests
from file_io import FileIO
base_path = "https://graph.facebook.com"
token = "1825924290976649|f4a421b77888587f351418a5aa84762c"
page_ids = ['TheEconomist']

for page_id in page_ids:
    texts = []
    feedRequestUrl = base_path + "/" + page_id + "/feed?access_token=" + token

    def do(url):
        r = requests.get(url)
        json_dict = r.json()
        for data in json_dict['data']:
            texts.append(data['message'].replace('\n', ' '))
        print(len(texts))
        if len(texts) == 1000:
            return
        if 'paging' in json_dict.keys():
            do(url)
    do(feedRequestUrl)
    # write text
    FileIO.write_lines_to_file('./fb_posts/'+page_id+'.txt', texts)
