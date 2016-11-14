from bs4 import BeautifulSoup
import requests

def scrape1000Words(url):
    '''
    INPUT: url
    OUTPUT: list of words

    Takes the url, scrapes the info, and returns a list of the 1000 most common SAT words.
    '''
    r = requests.get(url).text
    bs = BeautifulSoup(r, 'html')
    contents = bs.select('span.TermText.qWord.lang-en')
    words = []
    for c in contents:
        words.append(c.text.strip())
    return words

def extract5000Words(filename):
    '''
    INPUT: filename (string)
    OUTPUT: list of words

    Takes the filename and parses through each line to get the words only.
    '''
    words = []
    with open(filename, 'r') as f:
        for line in f:
            words.append(line.split()[0])
    return words

def writeToFile(words, path):
    '''
    INPUT: list of words
    OUTPUT: None

    Takes in the list of words and writes each word to a separate line in a new file (given a specified path).
    '''
    with open(path, 'w') as f:
        for word in words:
            try:
                f.write(word+'\n')
            except UnicodeEncodeError:
                f.write(word.encode('utf-8')+'\n')

if __name__ == '__main__':
    # Scrape 1000 words, then save in new file
    words_1000 = scrape1000Words('https://quizlet.com/1026577/1000-most-common-sat-words-flash-cards/')
    writeToFile(words_1000, '../data/other/1000_words.txt')

    # Extract the 5000 words in existing file, then save in new file
    words_5000 = extract5000Words('../data/other/5000_words_raw.txt')
    writeToFile(words_5000, '../data/other/5000_words.txt')
