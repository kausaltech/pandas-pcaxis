from datetime import datetime

import requests

from .px_reader import PxParser


class PXWebAPI:
    def __init__(self, base_url, language='en', base_path='api/v1', data_method='get'):
        self.base_url = base_url
        self.language = language
        self.base_path = base_path
        self.data_method = data_method

    def get(self, path):
        url = '%s/%s/%s/%s' % (self.base_url, self.base_path, self.language, path)
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def post(self, path, params):
        url = '%s/%s/%s/%s' % (self.base_url, self.base_path, self.language, path)
        resp = requests.post(url, json=params)
        resp.raise_for_status()
        return resp

    def list_databases(self):
        return self.get('')

    def list_topics(self, dbid):
        topics = self.get(dbid)
        for topic in topics:
            if 'updated' in topic:
                topic['updated'] = datetime.fromisoformat(topic['updated'])
        return topics

    def get_raw_table(self, path):
        if self.data_method == 'post':
            resp = self.post(path, params={'query': [], 'response': {'format': 'px'}})
        else:
            url = '%s/Resources/PX/Databases/%s' % (self.base_url, path)
            resp = requests.get(url)
        resp.raise_for_status()
        return resp.content

    def get_table(self, path, as_df=False):
        parser = PxParser()
        pxfile = parser.parse(self.get_raw_table(path))
        return pxfile


if __name__ == '__main__':
    # api = PXWebAPI('http://api.aluesarjat.fi', 'fi')
    # p = 'Ympäristötilastot/12_Ymparistotalous/1_Taloudelliset%20tunnusluvut/T1_talousluvut.px'

    api = PXWebAPI('http://pxnet2.stat.fi/PXWeb', 'fi')
    p = 'StatFin/asu/asas/statfin_asas_pxt_116b.px'
    print(api.get_table(p))
