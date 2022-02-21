import fsspec, requests
from bs4 import BeautifulSoup
from skimage import io

def getFilesHttp(url: str,ext: str) -> list:
    def listFD(url, ext=''):
        page = requests.get(url).text
        # print(page)
        soup = BeautifulSoup(page, 'html.parser')
        return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    
    files = []
    for file in listFD(url, ext):
        files.append(file)
        
    return files

def getImage(fileObj):
    with fileObj as f:
        print('Reading {} \n'.format(f))
        return io.imread(f)

url = 'https://download.brainimagelibrary.org/df/75/df75626840c76c15/mouseID_362188-191815/CH1_0.35_100um/'
files = getFilesHttp(url, "tif")

for fileObj in files:
    image = getImage(fileObj)
    print(image.shape)